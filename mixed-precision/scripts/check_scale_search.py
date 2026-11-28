"""Evaluate RTN, unweighted grid and activation-weighted grid under both objectives."""
import json

import numpy as np

from export_model import ROOT, SOURCE, read_index
from verify_quantization import check_pair


def payload(file, desc):
    file.seek(desc.offset)
    return file.read(desc.nbytes)


def main():
    folder = ROOT / "results/models"
    models = ROOT / ".work/models"
    files = {k: models / f"{k}.bin" for k in ("dec_cross", "dec_cross_grid", "dec_cross_cal")}
    grid_check = check_pair(files["dec_cross"], files["dec_cross_grid"])
    descs = {d.name: d for d in read_index(files["dec_cross"])[1]}
    source = {d.name: d for d in read_index(SOURCE)[1]}
    moments = np.array(json.loads((folder / "calibration.json").read_text())["mean_square"])
    metrics = {}
    with SOURCE.open("rb") as fp16:
        for key, path in files.items():
            losses = {"sse": 0.0, "activation_weighted_sse": 0.0}
            with path.open("rb") as f:
                for name in grid_check["changed_tensors"]:
                    x = np.frombuffer(payload(fp16, source[name]), "<f2").astype(np.float32).reshape(-1, 768)
                    blocks = np.frombuffer(payload(f, descs[name]), "u1").reshape(-1, 18)
                    scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
                    nibbles = blocks[:, 2:]
                    levels = np.hstack((nibbles & 15, nibbles >> 4)).astype(np.int8) - 8
                    error = (x - (scales * levels).reshape(x.shape)) ** 2
                    losses["sse"] += float(np.sum(error, dtype=np.float64))
                    losses["activation_weighted_sse"] += float(np.sum(error * moments))
            metrics[key] = losses
    for winner, field in (("dec_cross_grid", "sse"), ("dec_cross_cal", "activation_weighted_sse")):
        if metrics[winner][field] > min(m[field] for m in metrics.values()) + 1e-8:
            raise ValueError(f"Exported bytes do not minimize the specified grid objective for {winner}")
    (folder / "grid-control.json").write_text(json.dumps({
        "rtn_grid_payload_check": grid_check, "reconstruction": metrics,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
