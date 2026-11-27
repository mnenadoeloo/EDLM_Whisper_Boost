import argparse, hashlib, io, random
from collections import Counter
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

SEED = 42
LIBRI = "openslr/librispeech_asr"
GOLOS = "bond005/sberdevices_golos_100h_farfield"

def stream(repo, config, split):
    ds = load_dataset(repo, config, split=split, streaming=True)
    return ds.cast_column("audio", Audio(decode=False))

def parse(row):
    text = (row.get("text") or row.get("transcription") or "").strip()
    if not text:
        return None
    raw = row["audio"]["bytes"]
    info = sf.info(io.BytesIO(raw))
    if info.duration > 30:
        return None
    return {"raw": raw, "text": " ".join(text.split()),
            "id": row.get("id") or "golos-" + hashlib.sha1(raw).hexdigest()[:16],
            "speaker": row.get("speaker_id"), "hash": hashlib.sha1(raw).hexdigest()}

def reservoir(ds, k):
    rng, sample, seen = random.Random(SEED), [], 0
    for row in ds:
        item = parse(row)
        if item is None:
            continue
        seen += 1
        if len(sample) < k:
            sample.append(item)
        elif (j := rng.randrange(seen)) < k:
            sample[j] = item
    assert len(sample) == k, f"only {len(sample)} usable rows, need {k}"
    rng.shuffle(sample)
    return sample

def from_train(ds, k, max_per_speaker=2):
    sample, speakers = [], Counter()
    for row in ds.shuffle(seed=SEED, buffer_size=1000):
        item = parse(row)
        if item is None:
            continue
        if item["speaker"] is not None:
            if speakers[item["speaker"]] >= max_per_speaker:
                continue
            speakers[item["speaker"]] += 1
        sample.append(item)
        if len(sample) == k:
            return sample
    raise SystemExit(f"train stream ended with {len(sample)} of {k} rows")

def save(items, folder, subset, lang):
    folder.mkdir(parents=True, exist_ok=True)
    for it in items:
        wav, sr = sf.read(io.BytesIO(it.pop("raw")), dtype="int16")
        assert sr == 16000, f"unexpected sample rate {sr}"
        path = folder / f"{it['id']}.wav"
        sf.write(path, wav, sr, subtype="PCM_16")
        it.update(path=str(path), subset=subset, lang=lang)
    print(f"{subset}: {len(items)} files", flush=True)
    return items

def write_manifest(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write("\t".join([it["path"], it["text"], it["id"], it["subset"], it["lang"]]) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).resolve()

    dev_en = (save(reservoir(stream(LIBRI, "clean", "validation"), 150), out / "dev_en", "dev_en_clean", "en")
              + save(reservoir(stream(LIBRI, "other", "validation"), 150), out / "dev_en", "dev_en_other", "en"))
    dev_ru = save(reservoir(stream(GOLOS, "default", "validation"), 500), out / "dev_ru", "dev_ru", "ru")
    cal_en = save(from_train(stream(LIBRI, "clean", "train.100"), 100), out / "calib", "calib_en", "en")
    cal_ru = save(from_train(stream(GOLOS, "default", "train"), 100), out / "calib", "calib_ru", "ru")

    calib = [it for pair in zip(cal_en, cal_ru) for it in pair]
    overlap = {i["hash"] for i in calib} & {i["hash"] for i in dev_en + dev_ru}
    assert not overlap, f"{len(overlap)} files are in both calibration and dev"

    write_manifest(out / "dev_en.tsv", dev_en)
    write_manifest(out / "dev_ru.tsv", dev_ru)
    write_manifest(out / "calib_ids.tsv", calib)
    (out / "calib.txt").write_text("".join(f"{i['path']}\t{i['lang']}\n" for i in calib))
    print("done:", out)

if __name__ == "__main__":
    main()
