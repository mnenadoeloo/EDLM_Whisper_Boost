"""Score completed experiment blocks and select policies using dev data only."""
import argparse
import csv
import json
import math
import random
import re
import statistics
import unicodedata
from pathlib import Path
from collections import defaultdict

from benchmark import audio_context, config_floor, read_jsonl, sha256

GROUPS = ("enc_attn", "enc_ffn", "dec_self", "dec_cross", "dec_ffn")
LANGUAGES = ("en", "ru")
MIB = 1024 ** 2


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").replace("’", "'")
    # Preserve internal apostrophes in contractions, split hyphenated words.
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", text))


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, item in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (item != actual)))
        previous = current
    return previous[-1]


def score(row):
    ref, hyp = map(normalize, (row["reference"], row["prediction"]))
    words = ref.split()
    chars = ref.replace(" ", "")
    return {"word_errors": distance(words, hyp.split()), "ref_words": len(words),
            "char_errors": distance(chars, hyp.replace(" ", "")), "ref_chars": len(chars)}


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = math.floor(index)
    return values[left] + (values[math.ceil(index)] - values[left]) * (index - left)


def aggregate(rows, size, scores, repeat_scores=None):
    first = [r for r in rows if r["repeat"] == 1]
    observations = ([scores[r["id"]] for r in first] if repeat_scores is None else
                    [repeat_scores[r["id"], r["prediction"]] for r in rows])
    counts = {key: sum(s[key] for s in observations) for key in observations[0]}
    duration = sum(r["duration_s"] for r in rows)
    wall = [r["wall_s"] for r in rows]
    repeats = sorted({r["repeat"] for r in rows})
    rtfs = [sum(r["wall_s"] for r in rows if r["repeat"] == rep) /
            sum(r["duration_s"] for r in first) for rep in repeats]
    return {**counts, "wer": 100 * counts["word_errors"] / counts["ref_words"],
            "cer": 100 * counts["char_errors"] / counts["ref_chars"], "model_mib": size / MIB,
            "clips": len(first), "audio_s": sum(r["duration_s"] for r in first),
            "repeats": len(repeats), "rtf": sum(wall) / duration,
            "rtf_repeat_min": min(rtfs), "rtf_repeat_max": max(rtfs),
            "p50_s": quantile(wall, 0.5), "p95_s": quantile(wall, 0.95),
            "peak_rss_mib": max(r["peak_rss_bytes"] for r in rows) / MIB,
            "mean_encode_s": statistics.mean(r["stages_ms"]["encode"] for r in rows) / 1000,
            "mean_decode_s": statistics.mean(r["stages_ms"]["decode"] for r in rows) / 1000,
            "mean_load_s": statistics.mean(r["stages_ms"]["load"] for r in rows) / 1000}


def load_results(raw_dir, split, configs=None, subset_ids=None, *, score_repeats=False):
    summary, scored = {}, {}
    manifest = {s["id"]: s for s in read_jsonl(Path(f"data/{split}.jsonl"))}
    manifest_hash = sha256(Path(f"data/{split}.jsonl"))
    required = set(manifest) if subset_ids is None else set(subset_ids)
    if not required or not required <= set(manifest):
        raise ValueError("Subset contains no clips or unknown IDs")
    reference_spec = None
    for path in sorted(raw_dir.glob(f"{split}-*.jsonl")):
        config = path.stem.split("-", 1)[1]
        if configs is not None and config not in configs:
            continue
        rows = read_jsonl(path)
        if subset_ids is not None:
            rows = [r for r in rows if r["id"] in required]
        if not rows:
            raise ValueError(f"No measured clips for {path}")
        counts = {(r["repeat"], r["id"]) for r in rows}
        target = {(rep, key) for rep in range(1, max(r["repeat"] for r in rows) + 1) for key in required}
        if counts != target or len(counts) != len(rows):
            raise ValueError(f"Incomplete block or duplicate measurement: {path}")
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["config"] != config or meta["split"] != split:
            raise ValueError(f"Inconsistent metadata: {path}")
        if meta["manifest_sha256"] != manifest_hash or meta["audio_sha256"] != {
            key: sample["wav_sha256"] for key, sample in manifest.items()
        }:
            raise ValueError(f"Different audio/manifest identity: {path}")
        expected_rule = "min(1500,max(256,64*ceil((duration_s+1)*50/64)))"
        floor = config_floor(config)
        if floor:
            expected_rule = expected_rule.replace("max(256", f"max({floor}")
        if meta["context_rule"] != expected_rule:
            raise ValueError(f"Unexpected audio context rule: {path}")
        run_spec = {key: value for key, value in meta.items()
                    if key not in {"config", "model_bytes", "model_sha256", "context_rule"}}
        if reference_spec is None:
            reference_spec = run_spec
        elif run_spec != reference_spec:
            raise ValueError(f"Different runtime, hardware or decoding options: {path}")
        first = {r["id"]: r for r in rows if r["repeat"] == 1}
        for r in rows:
            sample = manifest[r["id"]]
            ctx = audio_context(r["duration_s"], floor) if floor else 0
            if r["audio_ctx"] != ctx:
                raise ValueError(f"Incorrect audio context: {path} {r['id']}")
            if (r["config"], r["split"]) != (config, split) or abs(r["duration_s"] - sample["duration_s"]) > 0.001:
                raise ValueError(f"Incorrect run label or audio duration: {path} {r['id']}")
            if (r["reference"], r["language"]) != (sample["reference"], sample["language"]):
                raise ValueError(f"Manifest mismatch: {path} {r['id']}")
            if not score_repeats and r["prediction"] != first[r["id"]]["prediction"]:
                raise ValueError(f"Nondeterministic prediction: {path} {r['id']}")
        scored[config] = {key: score(row) for key, row in first.items()}
        repeat_scores = None
        if score_repeats:
            repeat_scores = {(k, r["prediction"]): scored[config][k] for k, r in first.items()}
            for r in rows:
                pair = r["id"], r["prediction"]
                if pair not in repeat_scores:
                    repeat_scores[pair] = score(r)
        for lang in (*LANGUAGES, "all"):
            subset = [r for r in rows if lang == "all" or r["language"] == lang]
            summary[config, lang] = aggregate(subset, meta["model_bytes"], scored[config], repeat_scores)
    if configs is not None and set(configs) != set(scored):
        raise ValueError(f"Missing configurations: {set(configs) - set(scored)}")
    return summary, scored


def sensitivity(summary):
    rows = []
    for group in GROUPS:
        delta_size = summary[group, "all"]["model_mib"] - summary["q4_0", "all"]["model_mib"]
        row = {"group": group, "delta_mib": delta_size,
               "delta_rtf": summary[group, "all"]["rtf"] - summary["q4_0", "all"]["rtf"]}
        for lang in LANGUAGES:
            for metric in ("wer", "cer"):
                row[f"delta_{metric}_{lang}_pp"] = summary[group, lang][metric] - summary["q4_0", lang][metric]
        mean_gain = -statistics.mean(row[f"delta_wer_{lang}_pp"] for lang in LANGUAGES)
        rows.append({**row, "macro_wer_gain_pp_per_mib": mean_gain / delta_size})
    return sorted(rows, key=lambda r: -r["macro_wer_gain_pp_per_mib"])


def select(summary, configs):
    def excess(cfg):
        return max(summary[cfg, lang]["wer"] - summary["fp16", lang]["wer"] for lang in LANGUAGES)
    eligible = [cfg for cfg in configs if excess(cfg) <= 1.0 + 1e-10]
    candidates = eligible or configs
    winner = min(candidates, key=lambda cfg: (0 if eligible else excess(cfg), summary[cfg, "all"]["rtf"],
                                               summary[cfg, "all"]["model_mib"], cfg))
    return {"config": winner, "meets_dev_budget": bool(eligible),
            "max_delta_wer_pp": excess(winner), "eligible": sorted(eligible)}


def paired_wer(scores, model, baseline, ids):
    # Percentile interval from paired utterance resamples, corpus WER in each resample.
    rng = random.Random(20261121)
    diffs = [(scores[model][k]["word_errors"] - scores[baseline][k]["word_errors"],
              scores[model][k]["ref_words"]) for k in ids]
    draws = []
    for _ in range(2000):
        sample = rng.choices(diffs, k=len(diffs))
        draws.append(100 * sum(d for d, _ in sample) / sum(n for _, n in sample))
    return quantile(draws, 0.025), quantile(draws, 0.975)


def paired_speed(time_maps, config, baseline, ids_by_lang):
    # Duration cancels in a paired RTF ratio. Average repetitions per clip first.
    groups = [[(time_maps[baseline][k], time_maps[config][k]) for k in ids] for ids in ids_by_lang]
    rng = random.Random(20261121)
    estimates = []
    for _ in range(2000):
        pairs = [p for group in groups for p in rng.choices(group, k=len(group))]
        estimates.append(sum(a for a, _ in pairs) / sum(b for _, b in pairs))
    return quantile(estimates, .025), quantile(estimates, .975)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        for row in rows:
            out.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})


def pareto(summary, cfg, configs):
    def objectives(name):
        return [summary[name, lang]["wer"] for lang in LANGUAGES] + [
            summary[name, "all"][metric] for metric in ("rtf", "model_mib")]
    other = [objectives(x) for x in configs if x != cfg]
    target = objectives(cfg)
    return not any(all(a <= b for a, b in zip(point, target)) and point != target for point in other)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["smoke", "dev", "test", "m3_dev", "m3_test", "m3_panel"])
    parser.add_argument("--summary-only", action="store_true",
                        help="write per-language metrics for the measured models without pilot policy selection")
    parser.add_argument("--configs", nargs="+", help="score only this completed experiment stage")
    parser.add_argument("--raw", type=Path, default=Path("results/raw"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.split not in ("dev", "test") and not args.summary_only:
        parser.error("Use --summary-only for a split outside the original dev/test pilot")
    args.out.mkdir(parents=True, exist_ok=True)
    summary, scores = load_results(args.raw, args.split, args.configs, score_repeats=args.summary_only)
    if args.summary_only:
        if not summary:
            parser.error(f"No results for {args.split} in {args.raw}")
        write_csv(args.out / f"{args.split}-summary.csv", [
            {"split": args.split, "config": cfg, "language": lang, **metrics}
            for (cfg, lang), metrics in summary.items()
        ])
        return
    configs = sorted(scores)
    rows = []
    for (cfg, lang), data in summary.items():
        rows.append({"split": args.split, "config": cfg, "language": lang, **data,
                     "speedup_vs_fp16": summary["fp16", lang]["rtf"] / data["rtf"],
                     "delta_wer_vs_fp16_pp": data["wer"] - summary["fp16", lang]["wer"],
                     "pareto": pareto(summary, cfg, configs)})
    write_csv(args.out / f"{args.split}-summary.csv", rows)
    if args.split == "dev":
        ranked = sensitivity(summary)
        write_csv(args.out / "sensitivity.csv", ranked)
        mixed = [cfg for cfg in configs if cfg in {*GROUPS, "encoder", "greedy"}]
        selection = {"greedy_top_two": [r["group"] for r in ranked[:2]],
                     "mixed_full": select(summary, mixed),
                     "all_full": select(summary, [c for c in configs if "+" not in c]),
                     "all_configs": select(summary, configs)}
        (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    else:
        samples = read_jsonl(Path("data/test.jsonl"))
        time_maps = {}
        for cfg in configs:
            timings = defaultdict(list)
            for row in read_jsonl(args.raw / f"test-{cfg}.jsonl"):
                timings[row["id"]].append(row["wall_s"])
            time_maps[cfg] = {key: statistics.mean(vals) for key, vals in timings.items()}
        ids_by_lang = [[s["id"] for s in samples if s["language"] == lang] for lang in LANGUAGES]
        contrasts = []
        for cfg in configs:
            controls = ["fp16"]
            if "+" in cfg:
                controls.append(cfg.split("+")[0])
                mode = cfg.split("+")[1]
                controls.extend([f"fp16+{mode}", f"q4_0+{mode}"])
                if cfg.endswith("+ctx768"):
                    controls.append(cfg.replace("+ctx768", "+ctx"))
            elif cfg not in ("q4_0", "fp16"):
                controls.append("q4_0")
            if cfg == "greedy":
                controls.append("q5_0")
            if cfg == "dec_cross_cal":
                controls.append("dec_cross")
            for base in dict.fromkeys(c for c in controls if c in configs and c != cfg):
                paired = {"config": cfg, "baseline": base,
                          "speedup": summary[base, "all"]["rtf"] / summary[cfg, "all"]["rtf"]}
                paired["speed_ci_low"], paired["speed_ci_high"] = paired_speed(time_maps, cfg, base, ids_by_lang)
                for lang, ids in zip(LANGUAGES, ids_by_lang):
                    lo, hi = paired_wer(scores, cfg, base, ids)
                    paired.update({f"delta_wer_{lang}_pp": summary[cfg, lang]["wer"] - summary[base, lang]["wer"],
                                   f"ci_low_{lang}_pp": lo, f"ci_high_{lang}_pp": hi})
                contrasts.append(paired)
        write_csv(args.out / "test-contrasts.csv", contrasts)


if __name__ == "__main__":
    main()
