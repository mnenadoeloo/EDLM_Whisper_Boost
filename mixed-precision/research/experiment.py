"""Reproduce the policy search and the preselected held-out comparisons on CPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from analyze import GROUPS


def run(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["dev", "test"])
    parser.add_argument("--out", type=Path, default=Path(".work/rerun"))
    parser.add_argument("--with-calibration", action="store_true", help="include dec_cross_cal versus dec_cross")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw"
    selection_file = args.out / "selection.json"

    def bench(split, configs, repeats):
        run("research/benchmark.py", split, *dict.fromkeys(configs), "--repeats", repeats, "--output", raw)

    def analyze(split, configs):
        run("research/analyze.py", split, "--raw", raw, "--out", args.out, "--configs", *dict.fromkeys(configs))

    if args.phase == "dev":
        bench("smoke", ["fp16", "q4_0", "q8_0", "enc_attn"], 1)
        configs = ["fp16", "q4_0", "q5_0", "q8_0", *GROUPS, "encoder", "fp16+ctx", "q4_0+ctx"]
        bench("dev", configs, 2)
        analyze("dev", configs)
        groups = json.loads(selection_file.read_text())["greedy_top_two"]
        policies = [json.loads(Path(f"policies/{name}.json").read_text()) for name in groups]
        greedy = dict(policies[0], rules=[rule for policy in policies for rule in policy["rules"]])
        policy_path = args.out / "greedy-policy.json"
        policy_path.write_text(json.dumps(greedy, indent=2) + "\n")
        run("scripts/export_model.py", policy_path, ".work/models/greedy.bin")
        bench("dev", ["greedy"], 2)
        configs.append("greedy")
        if args.with_calibration:
            bench("dev", ["dec_cross_grid", "dec_cross_cal"], 2)
            configs.extend(["dec_cross_grid", "dec_cross_cal"])
        analyze("dev", configs)
        mixed = json.loads(selection_file.read_text())["mixed_full"]["config"]
        contexts = [mixed + "+ctx", mixed + "+ctx768"]
        bench("dev", contexts, 2)
        analyze("dev", [*configs, *contexts])
    else:
        selection = json.loads(selection_file.read_text())
        mixed = selection["mixed_full"]["config"]
        finals = ["fp16", "q4_0", "q5_0", "q8_0", "greedy", mixed, "fp16+ctx", "q4_0+ctx", mixed + "+ctx",
                  mixed + "+ctx768", "fp16+ctx768", "q4_0+ctx768", selection["all_configs"]["config"]]
        if args.with_calibration:
            finals.extend(["dec_cross", "dec_cross_cal"])
        (args.out / "test-configs.json").write_text(json.dumps(list(dict.fromkeys(finals)), indent=2) + "\n")
        bench("test", finals, 3)
        analyze("test", finals)


if __name__ == "__main__":
    main()
