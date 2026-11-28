"""Record actual model payload sizes/storage types and file hashes across policies."""
import argparse
import csv
from pathlib import Path

from export_model import ROOT, TYPES, read_index, sha256


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / ".work/models")
    parser.add_argument("--output", type=Path, default=ROOT / "results/models")
    args = parser.parse_args()
    files = sorted(args.models.glob("*.bin"))
    if not files:
        parser.error("No checkpoints found")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    weights = {}
    for path in files:
        _, tensors = read_index(path)
        name = path.stem
        counts = {details[0]: 0 for details in TYPES.values()}
        for t in tensors:
            desc = TYPES[t.dtype][0]
            counts[desc] += 1
            row = weights.setdefault(t.name, {"tensor": t.name, "ggml_shape": "x".join(map(str, t.shape)),
                                               "eligible": int(t.eligible)})
            row[f"{name}_type"] = desc
            row[f"{name}_bytes"] = t.nbytes
        summaries.append({"model": name, "file_bytes": path.stat().st_size,
                          "sha256": sha256(path), **counts})
    write_csv(args.output / "checkpoints.csv", summaries)
    write_csv(args.output / "tensors.csv", list(weights.values()))


if __name__ == "__main__":
    main()
