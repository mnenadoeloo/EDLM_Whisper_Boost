# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Render stand-alone research figures from the scored CSVs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sensitivity(folder):
    data = rows(folder / "sensitivity.csv")
    fields = ["delta_wer_en_pp", "delta_wer_ru_pp", "delta_cer_en_pp", "delta_cer_ru_pp"]
    values = [[float(row[key]) for key in fields] for row in data]
    limit = max(0.1, max(abs(v) for row in values for v in row))
    fig, ax = plt.subplots(figsize=(8.3, 3.8), layout="constrained")
    colors = LinearSegmentedColormap.from_list("change", ["#4281af", "#fbfbf8", "#d38e53"])
    ax.imshow(values, cmap=colors, norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
    names = {"enc_attn": "Encoder · attention", "enc_ffn": "Encoder · FFN", "dec_ffn": "Decoder · FFN",
             "dec_self": "Decoder · self-attention", "dec_cross": "Decoder · cross-attention"}
    ax.set_yticks(range(len(data)), [names[r["group"]] for r in data])
    ax.set_xticks(range(4), ["WER · EN", "WER · RU", "CER · EN", "CER · RU"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=8)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:+.2f}" if value else "0.00", va="center", ha="center")
    fig.suptitle("Повышение одной группы Q4_0 → Q8_0", fontsize=15)
    ax.set_xlabel("Изменение к uniform Q4_0 на dev, п.п.     •     Меньше — лучше", labelpad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(folder / "sensitivity.png", dpi=170)
    plt.close(fig)


def test_comparison(folder):
    records = rows(folder / "test-summary.csv")
    indexed = {(r["config"], r["language"]): r for r in records}
    configs = sorted({r["config"] for r in records}, key=lambda c: -float(indexed[c, "all"]["rtf"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 6.4), sharey=True, layout="constrained",
                             gridspec_kw={"width_ratios": [1.7, 1]})
    ax, quality = axes
    y = range(len(configs))
    speeds = [float(indexed[c, "all"]["speedup_vs_fp16"]) for c in configs]
    selected = json.loads((folder / "selection.json").read_text())["all_configs"]["config"]
    colors_by_config = {c: "#407899" for c in configs}
    colors_by_config.update({"fp16": "#a4aeb2", selected: "#194761"})
    ax.barh(y, speeds, color=[colors_by_config[c] for c in configs], height=.58)
    ax.axvline(1, linewidth=1, linestyle=":", color="#293a40")
    for pos, val in zip(y, speeds):
        ax.text(val + max(speeds) * .018, pos, f"{val:.2f}×", va="center", fontsize=9)
    ax.set(xlim=(0, max(speeds) * 1.17), xlabel="Ускорение CLI относительно FP16")
    ax.set_yticks(list(y), configs)
    for label in ax.get_yticklabels():
        if label.get_text() == selected:
            label.set_fontweight("bold")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)

    langs = ["ru", "en"]
    values = [[float(indexed[c, lang]["delta_wer_vs_fp16_pp"]) for lang in langs] for c in configs]
    colors = LinearSegmentedColormap.from_list("wer_delta", ["#6ea3c2", "#f9f8f5", "#e4a565"])
    quality.imshow(values, cmap=colors, vmin=-10, vmax=10, aspect="auto")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            quality.text(j, i, f"{value:+.2f}" + (" *" if value > 1 else ""),
                         va="center", ha="center", fontsize=9)
    quality.set_xticks([0, 1], ["RU", "EN"])
    quality.xaxis.tick_top()
    quality.tick_params(length=0, pad=6)
    quality.set_xlabel("ΔWER к FP16, п.п.\n* выше бюджета +1 п.п.", labelpad=7)
    for spine in quality.spines.values():
        spine.set_visible(False)
    ax.set_ylim(len(configs) - .5, -.5)
    fig.suptitle("Скорость и точность на отложенных записях", fontsize=15)
    fig.savefig(folder / "test-comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=Path("results"))
    folder = parser.parse_args().folder
    sensitivity(folder)
    test_comparison(folder)
