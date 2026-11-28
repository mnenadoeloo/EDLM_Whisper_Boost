"""Build the disjoint M3 development, confirmation, and timing-panel manifests."""
import argparse
import json
import random

from prepare_audio import ROOT, SOURCES, convert, sha256, source_rows


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(split, entries):
    path = ROOT / "data" / f"{split}.jsonl"
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Frozen manifest differs: {path}")
    path.write_text(content)
    print(f"{split}: {len(entries)} recordings", flush=True)


def verify_audio():
    frozen = json.loads((ROOT / "results/m3/selection.json").read_text())["frozen_sha256"]
    for split in ("m3_dev", "m3_test", "m3_panel"):
        name = f"data/{split}.jsonl"
        manifest = ROOT / name
        if sha256(manifest.read_bytes()) != frozen[name]:
            raise ValueError(f"Frozen manifest differs: {manifest}")
        entries = load(manifest)
        for entry in entries:
            wav = ROOT / entry["audio"]
            if sha256(wav.read_bytes()) != entry["wav_sha256"]:
                raise ValueError(f"Audio checksum mismatch: {wav}")
        print(f"{split}: {len(entries)} verified recordings", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="check frozen manifests and local WAVs offline")
    if parser.parse_args().verify_only:
        verify_audio()
        return
    old = [r for split in ("calibration", "dev", "smoke", "test")
           for r in load(ROOT / "data" / f"{split}.jsonl")]
    seen_ids = {r["id"] for r in old}
    seen_sources = {r["source_audio_sha256"] for r in old}
    seen_wavs = {r["wav_sha256"] for r in old}
    speakers = {r["speaker_id"] for r in old if r["language"] == "en"}
    for split in ("dev", "test"):
        entries = []
        for lang in ("ru", "en"):
            source_split, size, _ = SOURCES[split, lang]
            rng = random.Random(f"whisper-m3-v1/{split}/{lang}")
            indices = list(range(size))
            if split == "dev":
                # Oversample metadata before filtering old speakers, then select by seeded rank.
                rng.shuffle(indices)
                indices = indices[:160]
            records = source_rows(split, lang, sorted(indices))
            if split == "dev":
                rank = {idx: i for i, idx in enumerate(indices)}
                records = sorted(records, key=lambda item: rank[item[0]])
            selected = []
            for idx, row in records:
                reference = row["text" if lang == "en" else "transcription"]
                if not reference or not reference.strip():
                    continue
                if lang == "en" and row["speaker_id"] in speakers:
                    continue
                if sha256(row["audio"]["bytes"]) in seen_sources:
                    continue
                entry, audio = convert(lang, source_split, idx, row)
                if entry["id"] in seen_ids or entry["wav_sha256"] in seen_wavs:
                    continue
                seen_ids.add(entry["id"])
                seen_sources.add(entry["source_audio_sha256"])
                seen_wavs.add(entry["wav_sha256"])
                dest = ROOT / entry["audio"]
                if not dest.exists() or sha256(dest.read_bytes()) != entry["wav_sha256"]:
                    temporary = dest.with_suffix(".wav.part")
                    temporary.write_bytes(audio)
                    temporary.replace(dest)
                selected.append(entry)
                if split == "dev" and len(selected) == 32:
                    break
            if split == "dev" and len(selected) != 32:
                raise ValueError("Too few eligible development recordings")
            entries.extend(selected)
        save("m3_" + split, entries)
        if split == "dev":
            speakers.update(r["speaker_id"] for r in entries if r["language"] == "en")
        else:
            panel = []
            for lang in ("ru", "en"):
                population = [r for r in entries if r["language"] == lang]
                panel.extend(random.Random("m3-panel/" + lang).sample(population, 32))
            save("m3_panel", panel)


if __name__ == "__main__":
    main()
