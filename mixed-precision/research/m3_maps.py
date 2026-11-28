"""Construct equal-payload candidate maps using measured GGML costs, without test data."""
import csv
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from benchmark import CLI, FLAGS, audio_context, read_jsonl, sha256
from analyze import write_csv

ARCHIVE = Path("results/m3")
OUT = Path(os.environ.get("WHISPER_M3_RESULTS", ARCHIVE))
RAW = OUT / "raw"
MODELS = Path(".work/models") if OUT == ARCHIVE else OUT / "models"
TYPES = ["q4_0", "q5_0", "q8_0"]
# A Q5 promotion adds 4 bytes/block, a Q8 promotion adds 16. Each role spans 12 layers.
UNITS = (0, 1, 4)
PARTS = ["query", "key", "value", "out"]
FAMILIES = [f"{stack}.blocks.*.{kind}.{part}.weight" for stack, kind in
            [("encoder", "attn"), ("decoder", "cross_attn")] for part in PARTS]
HEAD = "decoder.token_embedding.weight"
ROLE = ["enc_attn"] * 4 + ["cross_token", "cross_kv", "cross_kv", "cross_token", "head"]
N_FAMILY = 12 * 768 * 768
N_HEAD = 768 * 51865


def decode_steps():
    counts = []
    for row in read_jsonl(Path("data/calibration.jsonl")):
        args = [str(CLI), *(x for x in FLAGS if x != "-otxt"), "-t", "4", "-ac",
                str(audio_context(row["duration_s"], 768)), "-m", ".work/models/q4_0.bin",
                "-l", row["language"], "-f", row["audio"]]
        log = subprocess.run(args, text=True, capture_output=True, check=True).stderr
        def count(stage):
            return int(re.search(rf"{stage} time\s*=.*?/\s*(\d+) runs", log).group(1))
        if count("encode") != 1 or count("batchd") != 4:
            raise ValueError("Expected one encoder pass and one four-token prompt")
        counts.append({"id": row["id"], "decode_steps": count("decode")})
    write_csv(OUT / "calibration-steps.csv", counts)
    return statistics.mean(row["decode_steps"] for row in counts)


def propose():
    observations = defaultdict(list)
    with (OUT / "matmul.csv").open() as file:
        for row in csv.DictReader(file):
            observations[row["role"], row["format"], int(row["N"])].append(float(row["us"]))
    if len(observations) != 3 * 14 or any(len(series) != 21 for series in observations.values()):
        raise ValueError("Incomplete kernel measurements")
    med = {k: statistics.median(v) for k, v in observations.items()}
    steps = decode_steps()
    def estimate(role, fmt):
        if role in ("enc_attn", "cross_kv"):
            return med[role, fmt, 768]
        return med[role, fmt, 4] + steps * med[role, fmt, 1]
    # The Q4 costs of untouched roles are shared and cancel. Estimate only the varied matrices.
    costs = [[(1 if role == "head" else 12) * estimate(role, fmt) for fmt in TYPES] for role in ROLE]
    budget = 16 * N_FAMILY
    best = {}
    for types in itertools.product(range(3), repeat=8):
        enc_units = sum(UNITS[t] for t in types[:4])
        for head in (0, 1):
            allocated = N_FAMILY * sum(UNITS[t] for t in types) + UNITS[head] * N_HEAD
            if not 0 <= budget - allocated < N_FAMILY:
                continue
            if head == 0 and enc_units not in (0, 4, 8, 12, 16):
                continue
            key = "head" if head else f"enc{enc_units // 4}"
            fmt = (*types, head)
            prediction = sum(c[t] for c, t in zip(costs, fmt))
            if key not in best or (prediction, fmt) < best[key]:
                best[key] = prediction, fmt

    candidates = []
    for key, (us, types) in sorted(best.items()):
        rules = [{"pattern": name, "format": TYPES[t].upper(), "method": "rtn"}
                 for name, t in zip([*FAMILIES, HEAD], types) if t]
        policy = {"source_model": "openai/whisper-small", "default": {"format": "Q4_0", "method": "rtn"},
                  "rules": rules}
        name = "m3_" + key
        for existing in ("enc_attn", "dec_cross"):
            same = ((2, 2, 2, 2, 0, 0, 0, 0, 0) if existing == "enc_attn" else
                    (0, 0, 0, 0, 2, 2, 2, 2, 0))
            if types == same:
                name = existing
        policy_dir = OUT / "policies"
        policy_dir.mkdir(exist_ok=True)
        path = policy_dir / f"{name}.json"
        path.write_text(json.dumps(policy, indent=2) + "\n")
        target = Path(f".work/models/{name}.bin")
        if name.startswith("m3_"):
            MODELS.mkdir(parents=True, exist_ok=True)
            target = MODELS / f"{name}.bin"
            subprocess.run([sys.executable, "scripts/export_model.py", path, target], check=True)
        candidates.append({"model": name, "bucket": key, "profiled_roles_ms": us / 1000,
                           "formats": ",".join(TYPES[t] for t in types), "model_bytes": target.stat().st_size,
                           "sha256": sha256(target)})
    write_csv(OUT / "candidates.csv", candidates)


if __name__ == "__main__":
    propose()
