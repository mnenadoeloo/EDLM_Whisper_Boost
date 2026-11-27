import argparse, csv
from pathlib import Path

import numpy as np

from test_axis import read_tensors

def errors(w, wq, stats):
    sq = ((w.astype(np.float64) - wq) ** 2).sum(0)
    ref = (w.astype(np.float64) ** 2).sum(0)
    out = {"plain": sq.sum() / ref.sum()}
    for label, s in stats.items():
        out[label] = (s * sq).sum() / (s * ref).sum()
    return out

def gmean(x):
    return float(np.exp(np.mean(np.log(x))))

def self_check():
    w = np.array([[1.0, 2.0], [1.0, 2.0]])
    wq = np.array([[1.0, 1.0], [1.0, 2.0]])
    r = errors(w, wq, {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    assert np.isclose(r["plain"], 1 / 10) and r["a"] == 0 and np.isclose(r["b"], 1 / 8)

def main():
    self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--f16", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="e.g. A.bin B.bin C.bin")
    ap.add_argument("--stats", nargs="+", required=True, help="label=file.npz ...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(a.models) >= 3, "need at least A B C"

    stats = {label: dict(np.load(path)) for label, path in (x.split("=", 1) for x in a.stats)}
    layers = sorted(next(iter(stats.values())))
    assert all(sorted(v) == layers for v in stats.values()), "stats files cover different layers"

    orig = read_tensors(a.f16, layers)
    names = [Path(m).stem for m in a.models]
    res = {}
    for path, name in zip(a.models, names):
        q = read_tensors(path, layers)
        for layer in layers:
            r = errors(orig[layer], q[layer], {k: v[layer] for k, v in stats.items()})
            assert r["plain"] < 0.09, f"{name} {layer}: relative error {r['plain']:.3f} is too big"
            res[name, layer] = r

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = ["plain", *stats]
    with open(out / "layers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "model", *metrics])
        w.writerows([layer, n, *(f"{res[n, layer][m]:.8g}" for m in metrics)] for layer in layers for n in names)

    base, b, c = names[0], names[-2], names[-1]
    lines = []
    for group in ["encoder", "decoder", ""]:
        sel = [x for x in layers if x.startswith(group)]
        if not sel:
            continue
        lines.append(f"\n== {group or 'all'} ({len(sel)} layers): geometric mean of per-layer error ratios (<1 is better)")
        for m in metrics:
            ratios = [f"{n}/{base}={gmean([res[n, x][m] / res[base, x][m] for x in sel]):.3f}" for n in names[1:]]
            cb = [res[c, x][m] / res[b, x][m] for x in sel]
            lines.append(f"  {m:>8}: " + "  ".join(ratios) +
                         f"  {c}/{b}={gmean(cb):.3f}  ({c} better than {b} in {sum(r < 1 for r in cb)}/{len(sel)})")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")

if __name__ == "__main__":
    main()
