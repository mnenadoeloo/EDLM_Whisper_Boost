import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_model import Tensor, read_index, resolve_policy


def test_rules_apply_to_eligible_tensors_and_reject_overlap_and_skips():
    weights = [Tensor("encoder.blocks.0.attn.query.weight", (32, 32), 1, 0, 2048),
               Tensor("encoder.blocks.0.attn.query.bias", (32,), 0, 0, 128)]
    policy = json.loads((ROOT / "policies/enc_attn.json").read_text())
    resolved = resolve_policy(policy, weights)
    assert list(resolved.values()) == [(8, "rtn"), (0, "copy")]
    rule = policy["rules"][0]
    for bad_rules in [[rule, rule], [dict(rule, pattern="*bias")], [dict(rule, pattern="bad")],
                      [dict(rule, method="gptq")], [dict(rule, method="calibration_aware")]]:
        with pytest.raises(ValueError):
            resolve_policy({**policy, "rules": bad_rules}, weights)


def test_reader_checks_mixed_directory_and_payload_boundaries(tmp_path):
    name = b"encoder.blocks.0.attn.query.weight"
    tensor = struct.pack("<5I", 2, len(name), 8, 32, 1) + name + bytes(34)
    body = b"lmgg" + bytes(40) + struct.pack("<4I", 2002, 0, 0, 0) + tensor
    directory = b"WMP1" + struct.pack("<3I", 1, len(name), 8) + name
    path = tmp_path / "mixed.bin"
    path.write_bytes(directory + body)
    assert read_index(path)[1][0].nbytes == 34
    for invalid in (directory + body[:-1], directory[:-1], directory.replace(name, b"z" * len(name)) + body,
                    directory[:12] + struct.pack("<I", 6) + directory[16:] + body):
        path.write_bytes(invalid)
        with pytest.raises((ValueError, struct.error)):
            read_index(path)


def test_calibrated_q4_layout_and_weighted_objective():
    np = pytest.importorskip("numpy")
    from calibrated_q4 import pack_q4
    weights = np.linspace(-0.9, 0.9, 64, dtype=np.float16)
    moments = np.linspace(0.25, 2.0, 32)
    reference = np.full((2, 18), 0x88, dtype=np.uint8)
    reference[:, :2] = np.array([[0.125], [0.125]], dtype="<f2").view("u1")
    result, stats = pack_q4(weights.tobytes(), reference.tobytes(), moments)
    result = np.frombuffer(result, dtype="u1").reshape(-1, 18)
    scale = result[:, :2].copy().view("<f2").astype(np.float32)
    codes = np.hstack((result[:, 2:] % 16, result[:, 2:] // 16)).astype(np.int16) - 8
    calculated = np.sum((weights.reshape(2, 32) - scale * codes) ** 2 * moments)
    assert stats["calibrated_weighted_sse"] == pytest.approx(calculated)
    assert stats["calibrated_weighted_sse"] < stats["rtn_weighted_sse"]
    assert stats["changed_blocks"] == 2


def test_native_loader_rejects_truncated_directory_and_wrong_tensor_type(tmp_path):
    import subprocess

    cli = ROOT / ".work/whisper.cpp/build/bin/whisper-cli"
    baseline = ROOT / ".work/models/q4_0.bin"
    if not (cli.exists() and baseline.exists()):
        pytest.skip("Requires pinned CPU runtime and Q4_0 baseline")
    sample = ROOT / ".work/whisper.cpp/samples/jfk.wav"
    header_end, tensors = read_index(baseline)
    tensor = next(t for t in tensors if t.name == "encoder.blocks.0.attn.query.weight")
    name = tensor.name.encode()
    directory = b"WMP1" + struct.pack("<3I", 1, len(name), 8) + name
    with baseline.open("rb") as f:
        header = f.read(header_end)
    record = struct.pack("<5I", 2, len(name), tensor.dtype, *tensor.shape) + name
    invalid = [(directory[:-4], "invalid model data"),
               (directory + header + record, "storage type mismatch")]
    for blob, expected in invalid:
        path = tmp_path / "invalid.bin"
        path.write_bytes(blob)
        run = subprocess.run([cli, "-m", path, "-f", sample, "-ng"], capture_output=True, text=True)
        assert run.returncode != 0
        assert expected in run.stderr
