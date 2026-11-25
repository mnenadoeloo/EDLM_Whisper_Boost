"""Collect first-block Q/K/V diagonal activation moments on the disjoint train manifest."""
import argparse
import json
from pathlib import Path
import subprocess

from export_model import ROOT, SOURCE, SOURCE_SHA, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/calibration.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / ".work/calibration-recomputed.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if not rows or any(not row["source_split"].startswith("train") for row in rows):
        raise ValueError("Expected train calibration audio")
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("FP16 source hash mismatch")
    paths = [ROOT / r["audio"] for r in rows]
    if any(sha256(p) != r["wav_sha256"] for p, r in zip(paths, rows)):
        raise ValueError("Calibration audio hash mismatch")
    runtime = ROOT / ".work/whisper.cpp"
    ggml = runtime / "build/ggml/src"
    binary = ROOT / ".work/collect-qkv"
    subprocess.run(["c++", "-O2", "-std=c++17", '-DWHISPER_VERSION="1.8.2"',
                    "-I", runtime / "src", "-I", runtime / "examples",
                    "-I", runtime / "ggml/include", "-I", runtime / "include",
                    ROOT / "scripts/collect_qkv.cpp", runtime / "build/examples/libcommon.a",
                    "-L", ggml, "-lggml", "-lggml-cpu", "-lggml-base",
                    f"-Wl,-rpath,{ggml}", "-o", binary], check=True)
    numbers = subprocess.check_output([binary, SOURCE, *paths], text=True).splitlines()
    means = list(map(float, numbers[1:]))
    if len(means) != 768 or not all(0 < v < float("inf") for v in means):
        raise ValueError("Invalid activation moments")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "source_sha256": SOURCE_SHA, "manifest_sha256": sha256(args.manifest),
        "utterance_ids": [r["id"] for r in rows], "encoder_frames": int(numbers[0]),
        "tensor": "encoder.blocks.0.attn_ln output", "mean_square": means,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
