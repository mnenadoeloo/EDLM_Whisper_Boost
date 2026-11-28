"""Restore the fixed bilingual pilot from immutable Hugging Face Parquet shards."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import wave


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".work/data"
LIBRI = "openslr/librispeech_asr/resolve/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
GOLOS = "bond005/sberdevices_golos_100h_farfield/resolve/c93949f7140beef4adc404e7b54841e957f81c54"
# Original split, number of rows, pinned first (and, except for train, only) source shard.
SOURCES = {
    ("train", "en"): ("train.100", 28539, f"{LIBRI}/clean/train.100/0000.parquet"),
    ("train", "ru"): ("train", 9570, f"{GOLOS}/data/train-00000-of-00003-066abd799f2fa68d.parquet"),
    ("dev", "en"): ("validation", 2703, f"{LIBRI}/clean/validation/0000.parquet"),
    ("dev", "ru"): ("validation", 933, f"{GOLOS}/data/validation-00000-of-00001-fd7b65d8512dcdb5.parquet"),
    ("test", "en"): ("test", 2620, f"{LIBRI}/clean/test/0000.parquet"),
    ("test", "ru"): ("test", 1916, f"{GOLOS}/data/test-00000-of-00001-7fb58d4cda8ff9fe.parquet"),
}


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def convert(lang, split, index, row):
    source = row["audio"]["bytes"]
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        input=source, stdout=subprocess.PIPE, check=True,
    ).stdout
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    audio = buffer.getvalue()
    ident = f"ls-{row['id']}" if lang == "en" else f"golos-ff-{split}-{index:05d}"
    entry = dict(id=ident, language=lang, audio=f".work/data/{ident}.wav",
                 reference=row["text" if lang == "en" else "transcription"],
                 duration_s=len(pcm) / 32000, source_split=split, row_index=index,
                 source_audio_sha256=sha256(source), wav_sha256=sha256(audio))
    if lang == "en":
        entry.update(speaker_id=row["speaker_id"], chapter_id=row["chapter_id"])
    return entry, audio


def source_rows(split, lang, indices):
    import pyarrow.parquet as pq

    _, _, source = SOURCES[split, lang]
    name = "libri" if lang == "en" else "golos"
    shard = CACHE / f"{name}-{split}.parquet"
    if not shard.is_file():
        temporary = shard.with_suffix(".parquet.part")
        url = f"https://huggingface.co/datasets/{source}?download=true"
        subprocess.run(["curl", "-fsSL", "--retry", "3", "--retry-all-errors", url, "-o", str(temporary)], check=True)
        temporary.replace(shard)
    offset = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
        end = offset + batch.num_rows
        for index in indices:
            if offset <= index < end:
                yield index, batch.slice(index - offset, 1).to_pylist()[0]
        if max(indices) < end:
            break
        offset = end


def sample_indices(split, lang, size):
    if split == "smoke":
        return [0]
    if split == "calibration":
        return [118, 300, 703, 1015] if lang == "en" else [56, 304, 707, 1003]
    count = 24 if split == "test" else 8
    return [(2 * i + 1) * size // (2 * count) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifests", action="store_true", help="Recreate the manifest using the selection rule")
    splits = ["smoke", "dev", "test", "calibration"]
    parser.add_argument("--split", choices=splits)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in ([args.split] if args.split else splits):
        path = ROOT / "data" / f"{split}.jsonl"
        frozen = [] if args.write_manifests else [json.loads(line) for line in path.read_text().splitlines()]
        entries = []
        for lang in ("en", "ru"):
            partition = "train" if split in ("calibration", "smoke") else split
            source_split, size, _ = SOURCES[partition, lang]
            expected = {r["row_index"]: r for r in frozen if r["language"] == lang}
            # Select the center of equally sized bins in source order, before any ASR run.
            indices = sample_indices(split, lang, size)
            if not args.write_manifests and set(expected) != set(indices):
                raise ValueError(f"Manifest does not match the selection rule: {lang}/{split}")
            missing = []
            for index in indices:
                if index in expected:
                    existing = ROOT / expected[index]["audio"]
                    if existing.is_file() and sha256(existing.read_bytes()) == expected[index]["wav_sha256"]:
                        continue
                missing.append(index)
            if missing:
                for index, row in source_rows(partition, lang, missing):
                    entry, audio = convert(lang, source_split, index, row)
                    if index in expected and entry != expected[index]:
                        raise ValueError(f"Incorrect recording or transcript: {entry['id']}")
                    expected[index] = entry
                    destination = ROOT / entry["audio"]
                    temporary = destination.with_suffix(".wav.part")
                    temporary.write_bytes(audio)
                    temporary.replace(destination)
            entries.extend(expected[index] for index in indices)
        if args.write_manifests:
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries))
        print(f"{split}: {len(entries)} verified recordings", flush=True)


if __name__ == "__main__":
    main()
