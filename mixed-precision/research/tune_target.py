"""Measure offline CLI configurations on a target, select on dev and optionally confirm on test."""
import argparse
import fcntl
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from analyze import load_results, write_csv
from benchmark import BUILD, config_floor, device_identity, read_jsonl, sha256

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / ".work/models"
sys.path.insert(0, str(ROOT / "scripts"))
from export_model import SOURCE_SHA


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def prepare_models(names, directory, build):
    models = {}
    if names == ["all"]:
        names = sorted({p.stem for d in (MODELS, directory) for p in d.glob("*.bin")})
    for name in sorted(set(["fp16", *names])):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Use a checkpoint filename stem: {name}")
        path = directory / f"{name}.bin"
        if not path.exists():
            path = MODELS / path.name
        if not path.exists():
            policies = [ROOT / f"policies/{name}.json", ROOT / f"results/m3/policies/{name}.json"]
            if name == "greedy":
                policies.append(ROOT / "results/greedy-policy.json")
            policy = next((p for p in policies if p.is_file()), None)
            if not policy:
                raise ValueError(f"Provide {name}.bin in {directory} or prepare it first")
            ext = "dylib" if sys.platform == "darwin" else "so"
            path = directory / f"{name}.bin"
            command = [sys.executable]
            if any(rule["method"] != "rtn" for rule in json.loads(policy.read_text())["rules"]):
                command = ["uv", "run", "--with", "numpy==2.3.3", "python"]
            subprocess.run([*command, "scripts/export_model.py", policy, path, "--library",
                            build / f"ggml/src/libggml-base.{ext}"], check=True)
        models[name] = sha256(path)
    if models["fp16"] != SOURCE_SHA:
        raise ValueError("FP16 baseline differs from the original multilingual Whisper Small")
    return models


def measure(args, split, cases, profile):
    rows = []
    for threads in sorted({n for _, n in cases}):
        configs = sorted(c for c, n in cases if n == threads)
        raw = args.out / "raw" / f"t{threads}"
        subprocess.run([sys.executable, "research/benchmark.py", split, *configs, "--backend", args.backend,
                        "--threads", str(threads), "--repeats", str(args.repeats), "--output", raw,
                        "--overrides", args.model_dir], check=True)
        stats, _ = load_results(raw, split, configs, score_repeats=True)
        for cfg in configs:
            meta = json.loads((raw / f"{split}-{cfg}.json").read_text())
            if (meta["hardware"] != profile["device"] or meta["cli_sha256"] != profile["cli_sha256"]
                    or meta["model_sha256"] != profile["models_sha256"][cfg.split("+")[0]]):
                raise ValueError(f"Measured runtime or weights differ from the target profile: {cfg}")
            if args.backend != "cpu":
                proofs = [r["device_evidence"] for r in read_jsonl(raw / f"{split}-{cfg}.jsonl")]
                identities = {(p["backend"], p["device"], p["description"]) for p in proofs}
                if len(identities) != 1 or any(p["backend"] != args.backend or
                       not any(v > 0 for v in p["weight_buffers_mb"].values()) for p in proofs):
                    raise ValueError(f"Inconsistent GPU placement: {cfg}")
                key = dict(zip(("backend", "device", "description"), identities.pop()))
                active = args.out / "active-gpu.json"
                if active.exists() and json.loads(active.read_text()) != key:
                    raise ValueError("GPU changed between measurements")
                if not active.exists():
                    save_json(active, key)
            a, ru, en = (stats[cfg, lang] for lang in ("all", "ru", "en"))
            if a["repeats"] != args.repeats:
                raise ValueError(f"Unexpected repeat count: {cfg}")
            rows.append({"config": cfg, "model": cfg.split("+")[0], "context_floor": config_floor(cfg),
                         "threads": threads, "rtf": a["rtf"], "rtf_ru": ru["rtf"],
                         "rtf_en": en["rtf"], "wer_ru": ru["wer"], "wer_en": en["wer"],
                         "errors_ru": ru["word_errors"], "words_ru": ru["ref_words"],
                         "errors_en": en["word_errors"], "words_en": en["ref_words"],
                         "clips_ru": ru["clips"], "clips_en": en["clips"],
                         "p50_s": a["p50_s"], "p95_s": a["p95_s"], "model_mib": a["model_mib"],
                         "process_peak_rss_mib": a["peak_rss_mib"]})
    return rows


def relative(rows, reference_threads):
    base = next(r for r in rows if (r["config"], r["threads"]) == ("fp16", reference_threads))
    for r in rows:
        r["speedup"] = base["rtf"] / r["rtf"]
        r["delta_ru_pp"] = r["wer_ru"] - base["wer_ru"]
        r["delta_en_pp"] = r["wer_en"] - base["wer_en"]
        r["required_budget_pp"] = max(0., r["delta_ru_pp"], r["delta_en_pp"])


def best(rows, budget):
    eligible = (r for r in rows if r["required_budget_pp"] <= budget + 1e-10)
    return min(eligible, key=lambda r: (r["rtf"], r["required_budget_pp"], r["model_mib"], r["config"], r["threads"]))


def best_for_speed(points, speedup):
    return next((r for r in points if r["speedup"] + 1e-10 >= speedup), None)


def frontier(rows):
    # The smallest per-language WER budget at which each new fastest configuration becomes feasible.
    points = []
    for budget in sorted({0., *(r["required_budget_pp"] for r in rows)}):
        winner = best(rows, budget)
        if not points or (winner["config"], winner["threads"]) != (points[-1]["config"], points[-1]["threads"]):
            points.append({"budget_pp": budget, **winner})
    return points


def run(args):
    build = BUILD if args.backend == "cpu" else BUILD.with_name("build-" + args.backend)
    if not (build / "bin/whisper-cli").is_file():
        raise ValueError(f"Build the runtime first: uv run scripts/setup_runtime.py --backend {args.backend}")
    models = prepare_models(args.models, args.model_dir, build)
    configs = {"fp16"}
    for model in models:
        for ctx in args.contexts:
            cfg = model if ctx == "full" else model + ("+ctx" if ctx == "256" else "+ctx" + ctx)
            config_floor(cfg)
            configs.add(cfg)
    cases = [(c, n) for c in sorted(configs) for n in sorted(set(args.threads))]
    reference = args.threads[0]
    spec = {"workload": "one CLI process per recording, model load included", "device": device_identity(args.backend),
            "models_sha256": models, "cli_sha256": sha256(build / "bin/whisper-cli"), "split": args.dev,
            "manifest_sha256": sha256(Path(f"data/{args.dev}.jsonl")), "configs": sorted(configs),
            "threads": sorted(set(args.threads)), "reference": {"config": "fp16", "threads": reference},
            "repeats": args.repeats}
    profile = args.out / "profile.json"
    if profile.exists() and json.loads(profile.read_text()) != spec:
        raise ValueError(f"Hardware, runtime, data or search grid changed. Use a fresh --out directory: {args.out}")
    save_json(profile, spec)
    dev = measure(args, args.dev, cases, spec)
    relative(dev, reference)
    write_csv(args.out / "dev.csv", dev)
    points = frontier(dev)
    write_csv(args.out / "curve.csv", points)
    choices = [{"budget_pp": b, **best(dev, b)} for b in sorted(set(args.budget_pp))]
    targets = [{"min_speedup": s, "candidate": best_for_speed(points, s)} for s in sorted(set(args.min_speedup))]
    decision = {"selection_split": args.dev, "profile_sha256": sha256(profile), "dev_sha256": sha256(args.out / "dev.csv"),
                "reference": spec["reference"], "choices": choices, "speed_targets": targets}
    save_json(args.out / "selection.json", decision)
    for r in choices:
        print(f"dev budget {r['budget_pp']:g} pp: {r['config']} / {r['threads']} threads, {r['speedup']:.3f}x", flush=True)
    for target in targets:
        candidate = target["candidate"]
        label = f"{candidate['config']} / {candidate['threads']} threads" if candidate else "unreachable on dev"
        print(f"dev target {target['min_speedup']:g}x: {label}", flush=True)
    if args.confirm:
        dev_audio = {r["wav_sha256"] for r in read_jsonl(Path(f"data/{args.dev}.jsonl"))}
        if any(r["wav_sha256"] in dev_audio for r in read_jsonl(Path(f"data/{args.confirm}.jsonl"))):
            raise ValueError("Development and confirmation recordings overlap")
        winners = choices + [t["candidate"] for t in targets if t["candidate"]]
        selected = sorted({("fp16", reference), *((r["config"], r["threads"]) for r in winners)})
        held = measure(args, args.confirm, selected, spec)
        relative(held, reference)
        indexed = {(r["config"], r["threads"]): r for r in held}
        checks = []
        for r in choices:
            actual = indexed[r["config"], r["threads"]]
            checks.append({"budget_pp": r["budget_pp"], "meets_test_budget": actual["required_budget_pp"] <= r["budget_pp"] + 1e-10,
                           **actual})
        speed_checks = []
        for target in targets:
            candidate = target["candidate"]
            actual = indexed[candidate["config"], candidate["threads"]] if candidate else None
            speed_checks.append({"min_speedup": target["min_speedup"], "measurement": actual,
                                 "meets_test_speed": actual["speedup"] >= target["min_speedup"] if actual else None})
        save_json(args.out / f"confirmation-{args.confirm}.json", {"decision": decision, "test_split": args.confirm,
                  "manifest_sha256": sha256(Path(f"data/{args.confirm}.jsonl")), "measurements": held,
                  "choices": checks, "speed_targets": speed_checks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "metal", "cuda"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["fp16", "q4_0", "m3_enc1"], help="checkpoint stems, or all available .bin files")
    parser.add_argument("--model-dir", type=Path, default=MODELS, help="team checkpoint directory; falls back to .work/models")
    parser.add_argument("--contexts", nargs="+", default=["full", "768"], help="full and/or adaptive floors 1..1500")
    parser.add_argument("--threads", type=int, nargs="+", default=[4], help="first value is the FP16 reference thread count")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dev", choices=["dev", "m3_dev"], default="dev")
    parser.add_argument("--budget-pp", nargs="+", type=float, default=[0., 0.5, 1., 2.])
    parser.add_argument("--min-speedup", nargs="+", type=float, default=[], help="also select by a minimum speedup to FP16")
    parser.add_argument("--confirm", choices=["test", "m3_panel", "m3_test"], help="evaluate frozen choices on held-out recordings")
    args = parser.parse_args()
    if min(args.threads) < 1 or args.repeats < 1 or any(not math.isfinite(b) or b < 0 for b in args.budget_pp):
        parser.error("Use positive threads/repeats and finite, nonnegative budgets")
    if any(not math.isfinite(s) or s <= 0 for s in args.min_speedup):
        parser.error("--min-speedup requires positive, finite ratios")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".tune.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another tuning run is using {args.out}")
        run(args)


if __name__ == "__main__":
    main()
