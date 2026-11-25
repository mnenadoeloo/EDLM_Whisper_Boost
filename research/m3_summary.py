"""Verify the frozen M3 comparison and write per-language measurements and paired intervals."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from analyze import load_results, quantile, write_csv
from benchmark import m3_pilot, read_jsonl, sha256
from m3_maps import OUT, RAW


def paired_intervals(blocks):
    """Resample all four sums together. One block is a speaker (EN) or an utterance (RU)."""
    rng = random.Random(20261123)
    wer, speed = [], []
    for _ in range(2000):
        totals = list(map(sum, zip(*rng.choices(blocks, k=len(blocks)))))
        delta_errors, words, base_wall, new_wall = totals
        wer.append(100 * delta_errors / words)
        speed.append(base_wall / new_wall)
    return {"delta_wer_low_pp": quantile(wer, .025), "delta_wer_high_pp": quantile(wer, .975),
            "speedup_low": quantile(speed, .025), "speedup_high": quantile(speed, .975)}


def confirm(full):
    selection = json.loads((OUT / "selection.json").read_text())
    base, new = selection["pair"]
    for name, digest in selection["frozen_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    summaries, contrasts = [], []
    test_transcripts = {}
    stages = [("m3_test", 1), ("m3_panel", 3)] if full else [("m3_test", 1)]
    for split, repeats in stages:
        manifest = read_jsonl(Path(f"data/{split}.jsonl"))
        label = split if full else "m3_pilot"
        if not full:
            manifest = m3_pilot(manifest)
            (OUT / "pilot-ids.json").write_text(json.dumps([r["id"] for r in manifest], indent=2) + "\n")
        ids = {r["id"] for r in manifest}
        stats, scores = load_results(RAW, split, [base, new], subset_ids=None if full else ids)
        times = {}
        for cfg in (base, new):
            stem = RAW / f"{split}-{cfg}"
            metadata = json.loads(stem.with_suffix(".json").read_text())
            if metadata["model_sha256"] != selection["model_sha256"][cfg] or metadata["threads"] != 4:
                raise ValueError("Weight identity or thread count changed")
            times[cfg] = defaultdict(float)
            for r in read_jsonl(stem.with_suffix(".jsonl")):
                if r["id"] not in ids:
                    continue
                times[cfg][r["id"]] += r["wall_s"] / repeats
                key = (cfg, r["id"])
                if split == "m3_test":
                    test_transcripts[key] = r["prediction"]
                elif r["prediction"] != test_transcripts[key]:
                    raise ValueError(f"Transcript changed between panel and confirmation: {key}")
        for lang in ("ru", "en"):
            items = [r for r in manifest if r["language"] == lang]
            for cfg in (base, new):
                metrics = stats[cfg, lang]
                if metrics["repeats"] != repeats:
                    raise ValueError(f"Expected {repeats} repeats: {split} / {cfg}")
                summaries.append({"split": label, "config": cfg, "language": lang, **metrics})
            grouped = defaultdict(lambda: [0, 0, 0., 0.])
            for r in items:
                k = r["id"]
                a, b = scores[base][k], scores[new][k]
                g = grouped[r["speaker_id"] if lang == "en" else k]
                g[0] += b["word_errors"] - a["word_errors"]
                g[1] += a["ref_words"]
                g[2] += times[base][k]
                g[3] += times[new][k]
            contrasts.append({"split": label, "language": lang, "baseline": base, "candidate": new,
                              "bootstrap_unit": "speaker" if lang == "en" else "clip", "units": len(grouped),
                              "delta_wer_pp": stats[new, lang]["wer"] - stats[base, lang]["wer"],
                              "speedup": stats[base, lang]["rtf"] / stats[new, lang]["rtf"],
                              **paired_intervals(list(grouped.values()))})
    prefix = "" if full else "pilot-"
    write_csv(OUT / f"{prefix}summary.csv", summaries)
    write_csv(OUT / f"{prefix}contrasts.csv", contrasts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="require the entire M3 test and repeated panel")
    confirm(parser.parse_args().full)
