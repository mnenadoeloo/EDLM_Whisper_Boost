"""Run the registered development, freeze, and paired M3 confirmation from the repo root."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from analyze import LANGUAGES, load_results, write_csv
from benchmark import sha256
from m3_maps import ARCHIVE, MODELS, OUT, RAW

MODE = "+ctx768"
BASE = "enc_attn" + MODE


def run(split, configs, repeats, *, pilot=False):
    subprocess.run([sys.executable, "research/benchmark.py", split, *configs, "--repeats",
                    str(repeats), "--output", RAW, "--overrides", MODELS, *(["--pilot"] if pilot else [])], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the entire confirmation and three timing-panel repeats")
    args = parser.parse_args()
    with (OUT / "candidates.csv").open() as file:
        pool = {r["model"] + MODE: r for r in csv.DictReader(file)}
    configs = sorted({*pool, BASE, "fp16" + MODE, "q4_0" + MODE})
    run("m3_dev", configs, 1)
    dev, _ = load_results(RAW, "m3_dev", configs)
    write_csv(OUT / "dev-summary.csv", [{"config": cfg, "language": lang, **metrics}
              for (cfg, lang), metrics in dev.items()])
    excess = {cfg: max(dev[cfg, lang]["wer"] - dev[control, lang]["wer"]
                       for lang in LANGUAGES for control in (BASE, "fp16" + MODE))
              for cfg in pool if cfg.startswith("m3_")}
    valid = sorted(cfg for cfg, delta in excess.items() if delta <= 1.0 + 1e-10)
    winner = min(valid or excess, key=lambda cfg: (0 if valid else excess[cfg],
                 float(pool[cfg]["profiled_roles_ms"]), cfg))
    pair = [BASE, winner]
    models = {cfg: json.loads((RAW / f"m3_dev-{cfg}.json").read_text())["model_sha256"] for cfg in pair}
    if any(pool[cfg]["sha256"] != models[cfg] for cfg in pair):
        raise ValueError("Profiled and benchmarked weights differ")
    selection = {"pair": pair, "model_sha256": models, "meets_dev_budget": bool(valid),
                 "worst_dev_wer_delta_pp": excess, "eligible": valid,
                 "frozen_sha256": {str(p): sha256(p) for p in [*(Path("data") / f"m3_{s}.jsonl" for s in
                     ("dev", "test", "panel")), ARCHIVE / "PROTOCOL.md",
                     *(OUT / f for f in ("matmul.csv", "candidates.csv"))]}}
    path = OUT / "selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("Selection changed since freeze")
    path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Frozen pair {pair}, development budget met: {bool(valid)}", flush=True)
    run("m3_test", pair, 1, pilot=not args.full)
    if args.full:
        run("m3_panel", pair, 3)


if __name__ == "__main__":
    main()
