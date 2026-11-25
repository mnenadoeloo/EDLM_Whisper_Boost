"""Export a policy from the pinned FP16 GGML using whisper.cpp's GGML RTN kernels."""
import argparse
import ctypes as ct
import fnmatch
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"
SOURCE = ROOT / ".work/models/fp16.bin"
GGML_MAGIC = b"lmgg"
MIXED_MAGIC = b"WMP1"
# name, weight block length, block bytes (ggml.h / ggml-common.h)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18),
         6: ("Q5_0", 32, 22), 8: ("Q8_0", 32, 34)}
TYPE_ID = {info[0]: index for index, info in TYPES.items()}
# Model ftype and per-tensor ggml_type are distinct enums.
FTYPE = {"Q4_0": 2, "Q5_0": 8, "Q8_0": 7}
SKIP = {"encoder.conv1.bias", "encoder.conv2.bias",
        "encoder.positional_embedding", "decoder.positional_embedding"}


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    dtype: int
    offset: int
    nbytes: int

    @property
    def eligible(self):
        return len(self.shape) == 2 and self.name not in SKIP


def u32(stream):
    return struct.unpack("<I", stream.read(4))[0]


def read_index(path):
    """Return tensor payload offsets and the native header size up to weight records."""
    file_size = path.stat().st_size
    tensors = []
    with path.open("rb") as stream:
        magic = stream.read(4)
        overrides = {}
        if magic == MIXED_MAGIC:
            count = u32(stream)
            if not 1 <= count <= 8192:
                raise ValueError("Invalid mixed model directory size")
            for _ in range(count):
                length, dtype = u32(stream), u32(stream)
                if dtype not in (2, 6, 8) or not 1 <= length <= 256:
                    raise ValueError("Invalid mixed model descriptor")
                name = stream.read(length).decode("ascii")
                if len(name) != length or name in overrides:
                    raise ValueError("Truncated or duplicated mixed model descriptor")
                overrides[name] = dtype
            magic = stream.read(4)
        if magic != GGML_MAGIC:
            raise ValueError("Unrecognized GGML file")
        stream.seek(40, 1)
        model_ftype = u32(stream) % 1000
        mel, fft = struct.unpack("<2I", stream.read(8))
        stream.seek(mel * fft * 4, 1)
        for _ in range(u32(stream)):
            stream.seek(u32(stream), 1)
        header_end = stream.tell()
        while stream.tell() < file_size:
            ndim, length, dtype = struct.unpack("<3I", stream.read(12))
            if dtype not in TYPES or not 1 <= ndim <= 4 or not 0 < length <= 256:
                raise ValueError("Invalid tensor record")
            shape = struct.unpack(f"<{ndim}I", stream.read(4 * ndim))
            name = stream.read(length).decode("ascii")
            _, block, size = TYPES[dtype]
            if min(shape) <= 0 or shape[0] % block:
                raise ValueError(f"Invalid block alignment for {name}")
            nbytes = math.prod(shape) // block * size
            tensors.append(Tensor(name, shape, dtype, stream.tell(), nbytes))
            stream.seek(nbytes, 1)
        if stream.tell() != file_size or not tensors:
            raise ValueError("Truncated or empty weight file")
    if overrides:
        default = next((TYPE_ID[fmt] for fmt, ft in FTYPE.items() if ft == model_ftype), None)
        for tensor in tensors:
            if tensor.eligible and overrides.pop(tensor.name, default) != tensor.dtype:
                raise ValueError(f"Mixed directory disagrees with tensor {tensor.name}")
        if overrides:
            raise ValueError(f"Mixed directory refers to an unknown or skipped tensor {next(iter(overrides))}")
    return header_end, tensors


def resolve_policy(policy, tensors):
    required = {"source_model", "default", "rules"}
    if not required <= policy.keys() <= required | {"calibration"} or policy["source_model"] != "openai/whisper-small":
        raise ValueError("Expected source_model=openai/whisper-small, default and rules")
    default = policy["default"]
    rules = policy["rules"]
    if set(default) != {"format", "method"} or not isinstance(rules, list):
        raise ValueError("Invalid default or rules")
    for rule in rules:
        if set(rule) != {"pattern", "format", "method"} or not isinstance(rule["pattern"], str):
            raise ValueError(f"Invalid rule {rule}")
    if default["method"] != "rtn":
        raise ValueError("Default method must be rtn")
    for spec in [default, *rules]:
        if spec["format"] not in FTYPE or spec["method"] not in ("rtn", "grid_unweighted", "calibration_aware"):
            raise ValueError(f"Unsupported quantization spec {spec}")
    choices = {}
    for rule in rules:
        matches = [t for t in tensors if fnmatch.fnmatchcase(t.name, rule["pattern"])]
        if not matches:
            raise ValueError(f"Unmatched pattern {rule['pattern']}")
        for t in matches:
            if not t.eligible or t.shape[0] % 32 or t.name in choices:
                raise ValueError(f"Skipped, misaligned or multiply matched tensor {t.name}")
            if rule["method"] != "rtn" and (
                rule["format"] != "Q4_0" or t.name not in {
                    f"encoder.blocks.0.attn.{key}.weight" for key in ("query", "key", "value")
                }
            ):
                raise ValueError(f"No first-block Q4_0 activation statistics for {t.name}")
            choices[t.name] = rule
    selections = {t.name: (TYPE_ID[choices.get(t.name, default)["format"]],
                           choices.get(t.name, default)["method"]) if t.eligible else (t.dtype, "copy")
                  for t in tensors}
    needs_stats = any(method == "calibration_aware" for _, method in selections.values())
    if needs_stats != ("calibration" in policy):
        raise ValueError("calibration field must accompany calibration_aware rules")
    return selections


class GGML:
    """Call the reference C kernels from the same build used for inference."""
    def __init__(self, library):
        self.lib = ct.CDLL(str(library))
        self.lib.ggml_fp16_to_fp32_row.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64]
        self.lib.ggml_fp16_to_fp32_row.restype = None
        self.lib.ggml_quantize_chunk.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                                                 ct.c_int64, ct.c_int64, ct.c_int64, ct.c_void_p]
        self.lib.ggml_quantize_chunk.restype = ct.c_size_t

        class InitParams(ct.Structure):
            _fields_ = [("mem_size", ct.c_size_t), ("mem_buffer", ct.c_void_p), ("no_alloc", ct.c_bool)]

        self.lib.ggml_init.argtypes = [InitParams]
        self.lib.ggml_init.restype = ct.c_void_p
        self.lib.ggml_free.argtypes = [ct.c_void_p]
        self.lib.ggml_free.restype = None
        self.lib.ggml_free(self.lib.ggml_init(InitParams(0, None, False)))

    def quantize(self, data, tensor, dtype):
        n = math.prod(tensor.shape)
        if tensor.dtype == TYPE_ID["F16"]:
            floats = (ct.c_float * n)()
            self.lib.ggml_fp16_to_fp32_row(data, floats, n)
        else:
            floats = data  # F32 only; source is hash-pinned.
        _, block, nbytes = TYPES[dtype]
        packed = ct.create_string_buffer(n // block * nbytes)
        size = self.lib.ggml_quantize_chunk(dtype, floats, packed, 0,
                                            n // tensor.shape[0], tensor.shape[0], None)
        if size != len(packed):
            raise RuntimeError(f"GGML returned {size} bytes for {tensor.name}, expected {len(packed)}")
        return packed


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def export(source, policy, output, library):
    if source.resolve() == output.resolve():
        raise ValueError("The output must differ from the FP16 source")
    if sha256(source) != SOURCE_SHA:
        raise ValueError("Source SHA-256 differs from the pinned multilingual Small FP16 GGML")
    header_end, tensors = read_index(source)
    selections = resolve_policy(policy, tensors)
    types = {name: dtype for name, (dtype, _) in selections.items()}
    stats = None
    diagnostics = {}
    if "calibration" in policy:
        ref = policy["calibration"]
        path = ROOT / ref["path"]
        if sha256(path) != ref["sha256"]:
            raise ValueError("Calibration statistics checksum mismatch")
        stats = json.loads(path.read_text())
        if stats["source_sha256"] != SOURCE_SHA or len(stats["mean_square"]) != 768 or not all(
            0 < x < float("inf") for x in stats["mean_square"]
        ):
            raise ValueError("Invalid FP16 calibration statistics")
    default = policy["default"]["format"]
    overrides = {t.name: types[t.name] for t in tensors if t.eligible and types[t.name] != TYPE_ID[default]}
    ggml = GGML(library)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with source.open("rb") as original, tmp_path.open("wb") as dest:
            if overrides:
                dest.write(MIXED_MAGIC + struct.pack("<I", len(overrides)))
                for name, dtype in sorted(overrides.items()):
                    dest.write(struct.pack("<II", len(name), dtype) + name.encode("ascii"))
            header = bytearray(original.read(header_end))
            struct.pack_into("<I", header, 44, 2000 + FTYPE[default])
            dest.write(header)
            for tensor in tensors:
                dtype = types[tensor.name]
                name = tensor.name.encode("ascii")
                dest.write(struct.pack("<3I", len(tensor.shape), len(name), dtype))
                dest.write(struct.pack(f"<{len(tensor.shape)}I", *tensor.shape) + name)
                original.seek(tensor.offset)
                payload = original.read(tensor.nbytes)
                if dtype != tensor.dtype:
                    packed = ggml.quantize(payload, tensor, dtype)
                    method = selections[tensor.name][1]
                    if method != "rtn":
                        from calibrated_q4 import pack_q4
                        importance = stats["mean_square"] if method == "calibration_aware" else [1.0] * tensor.shape[0]
                        packed, diag = pack_q4(payload, packed, importance)
                        if method == "grid_unweighted":
                            diag["rtn_sse"] = diag.pop("rtn_weighted_sse")
                            diag["grid_sse"] = diag.pop("calibrated_weighted_sse")
                        diagnostics[tensor.name] = diag
                    payload = packed
                dest.write(payload)
        _, actual = read_index(tmp_path)
        if {t.name: t.dtype for t in actual} != types:
            raise ValueError("Exported tensor types differ from policy")
        os.replace(tmp_path, output)
    finally:
        tmp_path.unlink(missing_ok=True)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    ext = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--library", type=Path, default=ROOT / f".work/whisper.cpp/build/ggml/src/libggml-base.{ext}")
    parser.add_argument("--diagnostics", type=Path, help="Save per-matrix calibration reconstruction errors")
    args = parser.parse_args()
    diagnostics = export(args.source, json.loads(args.policy.read_text()), args.output, args.library)
    if args.diagnostics:
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"{args.output}: {args.output.stat().st_size} bytes; sha256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
