"""Compare stock uniform exports byte-for-byte and check the fixed-map method control."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from export_model import ROOT, SOURCE, read_index, sha256


def check_pair(first, second):
    _, left = read_index(first)
    _, right = read_index(second)
    if left != right or first.stat().st_size != second.stat().st_size:
        raise ValueError("Calibration-aware model changed tensor layout or byte size")
    changed = []
    with first.open("rb") as a, second.open("rb") as b:
        cursor = 0
        for tensor in left:
            if a.read(tensor.offset - cursor) != b.read(tensor.offset - cursor):
                raise ValueError(f"Model headers differ at {tensor.name}")
            size = tensor.nbytes
            if hashlib.sha256(a.read(size)).digest() != hashlib.sha256(b.read(size)).digest():
                changed.append(tensor.name)
            cursor = tensor.offset + size
    expected = {f"encoder.blocks.0.attn.{qkv}.weight" for qkv in ("query", "key", "value")}
    if set(changed) != expected:
        raise ValueError(f"Unexpected changed payloads: {changed}")
    return {"file_bytes": first.stat().st_size, "unchanged_tensors": len(left) - len(changed),
            "changed_tensors": changed, "rtn_sha256": sha256(first), "variant_sha256": sha256(second)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-pair", action="store_true")
    args = parser.parse_args()
    models = ROOT / ".work/models"
    out = ROOT / "results/models"
    if args.method_pair:
        result = check_pair(models / "dec_cross.bin", models / "dec_cross_cal.bin")
        (out / "method-control.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    with tempfile.TemporaryDirectory(dir=models.parent) as directory:
        uniform = {}
        for fmt in ("q4_0", "q5_0", "q8_0"):
            native = Path(directory) / f"{fmt}.bin"
            subprocess.run([ROOT / ".work/whisper.cpp/build/bin/quantize", SOURCE, native, fmt],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            native_sha = sha256(native)
            if sha256(models / f"{fmt}.bin") != native_sha:
                raise ValueError(f"{fmt} differs from stock quantize")
            uniform[fmt] = {"bytes": native.stat().st_size, "sha256": native_sha}
    (out / "stock-quantize.json").write_text(json.dumps(uniform, indent=2) + "\n")


if __name__ == "__main__":
    main()
