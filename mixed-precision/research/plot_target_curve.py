# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Draw measured dev trade-offs; each device is normalized to its own FP16 reference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-budget", type=float, default=2.)
    args = parser.parse_args()
    if args.max_budget <= 0:
        parser.error("--max-budget must be positive")
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    for folder in args.profiles:
        with (folder / "curve.csv").open() as f:
            rows = [r for r in csv.DictReader(f) if float(r["budget_pp"]) <= args.max_budget]
        profile = json.loads((folder / "profile.json").read_text())
        if not rows:
            raise ValueError(f"No feasible configuration in {folder}")
        budgets = [float(r["budget_pp"]) for r in rows]
        gains = [float(r["speedup"]) for r in rows]
        label = f"{folder.name} ({profile['device']['backend'].upper()})"
        line, = ax.step([*budgets, args.max_budget], [*gains, gains[-1]], where="post", label=label)
        ax.scatter(budgets, gains, color=line.get_color(), s=35, zorder=3)
    ax.set(xlabel="Допустимый рост WER на каждом языке, п.п.", ylabel="Ускорение к FP16 на том же устройстве, ×",
           xlim=(0, args.max_budget), title="Самая быстрая измеренная конфигурация в бюджете WER")
    ax.grid(alpha=.24)
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
