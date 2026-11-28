import argparse, os, struct, subprocess, sys, tempfile
import numpy as np
from safetensors import safe_open

F32, F16, Q4_1, Q5_1 = 0, 1, 3, 7
LAYERS = {
    "encoder.blocks.0.attn.query.weight": "model.encoder.layers.0.self_attn.q_proj.weight",
    "encoder.blocks.0.mlp.0.weight": "model.encoder.layers.0.fc1.weight",
    "encoder.blocks.0.mlp.2.weight": "model.encoder.layers.0.fc2.weight",
}
SPIKE_CHANNELS = [5, 200, 431, 700]
SPIKE = 1000.0

def decode_q5_1(raw):
    """Block: fp16 d, fp16 m, uint32 qh (5th bits), 16 bytes; value = q * d + m."""
    b = np.frombuffer(raw, dtype=[("d", "<f2"), ("m", "<f2"), ("qh", "<u4"), ("q", "u1", 16)])
    j = np.arange(16, dtype=np.uint32)
    qh = b["qh"][:, None]
    lo = (b["q"] & 15) | (((qh >> j) << 4) & 16)
    hi = (b["q"] >> 4) | ((qh >> (j + 12)) & 16)
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return q * b["d"].astype(np.float32)[:, None] + b["m"].astype(np.float32)[:, None]

def read_tensors(path, wanted):
    """Return {name: float32 array of shape (ne[1], ne[0])} from a whisper.cpp .bin file."""
    out = {}
    with open(path, "rb") as f:
        def i32():
            return struct.unpack("<i", f.read(4))[0]
        assert i32() == 0x67676D6C, "bad magic"
        f.seek(11 * 4, 1)                    
        n_mel, n_fft = i32(), i32()
        f.seek(n_mel * n_fft * 4, 1)          
        for _ in range(i32()):       
            f.seek(i32(), 1)
        while head := f.read(12):
            n_dims, name_len, ttype = struct.unpack("<3i", head)
            ne = [i32() for _ in range(n_dims)]
            name = f.read(name_len).decode()
            n = int(np.prod(ne))
            if ttype == F32:
                size = n * 4
            elif ttype == F16:
                size = n * 2
            elif ttype == Q4_1:
                size = n // 32 * 20
            elif ttype == Q5_1:
                size = n // 32 * 24
            else:
                sys.exit(f"unexpected tensor type {ttype} for {name}")
            if name not in wanted:
                f.seek(size, 1)
                continue
            raw = f.read(size)
            if ttype == Q4_1:
                b = np.frombuffer(raw, dtype=[("d", "<f2"), ("m", "<f2"), ("q", "u1", 16)])
                q = np.concatenate([b["q"] & 15, b["q"] >> 4], axis=1).astype(np.float32)
                x = q * b["d"].astype(np.float32)[:, None] + b["m"].astype(np.float32)[:, None]
            elif ttype == Q5_1:
                x = decode_q5_1(raw)
            else:
                x = np.frombuffer(raw, dtype="<f4" if ttype == F32 else "<f2").astype(np.float32)
            out[name] = x.reshape(ne[1], ne[0])
    missing = set(wanted) - set(out)
    assert not missing, f"not found in {path}: {missing}"
    return out

def write_imatrix(path, name, vec):
    with open(path, "wb") as f:
        key = name.encode()
        f.write(struct.pack("<i", 1) + struct.pack("<i", len(key)) + key)
        f.write(struct.pack("<i", len(vec)) + vec.astype("<f4").tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", required=True)
    ap.add_argument("--f16", required=True)
    ap.add_argument("--hf", required=True, help="local HF model dir with model.safetensors")
    a = ap.parse_args()

    orig = read_tensors(a.f16, LAYERS)
    ok = True

    print("1. orientation (ggml (ne1, ne0) vs HF [out, in])")
    with safe_open(os.path.join(a.hf, "model.safetensors"), framework="np") as st:
        for name, hf_name in LAYERS.items():
            hf = st.get_tensor(hf_name).astype(np.float32)
            same = orig[name].shape == hf.shape and np.allclose(orig[name], hf, atol=2e-3, rtol=2e-3)
            ok &= bool(same)
            print(f"  {'PASS' if same else 'FAIL'} {name} {orig[name].shape}")

    print("2. effect of spiked channels (column error: spike / ones, must be < 1)")
    with tempfile.TemporaryDirectory() as tmp:
        def quantize(out, *flags):
            subprocess.run([a.quantize, a.f16, out, "q4_1", *flags], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ones_path = os.path.join(tmp, "ones.bin")
        quantize(ones_path, "--imatrix-ones")
        ones = read_tensors(ones_path, LAYERS)

        for name in LAYERS:
            w = orig[name]
            vec = np.ones(w.shape[1], dtype=np.float32)
            vec[SPIKE_CHANNELS] = SPIKE
            imat, spike_path = os.path.join(tmp, "spike.imat"), os.path.join(tmp, "spike.bin")
            write_imatrix(imat, name, vec)
            quantize(spike_path, "--imatrix", imat)
            spiked = read_tensors(spike_path, LAYERS)[name]

            e_ones, e_spike = (ones[name] - w) ** 2, (spiked - w) ** 2
            rel = np.sqrt(e_ones.mean() / (w ** 2).mean())
            assert rel < 0.3, f"{name}: relative error {rel:.2f} is too big"

            ratios = e_spike[:, SPIKE_CHANNELS].sum(0) / e_ones[:, SPIKE_CHANNELS].sum(0)
            passed = bool((ratios < 0.9).all())
            ok &= passed
            line = f"  {'PASS' if passed else 'FAIL'} {name}: columns {np.round(ratios, 2).tolist()}"
            rows = [c for c in SPIKE_CHANNELS if c < w.shape[0]]
            row_ratios = e_spike[rows].sum(1) / e_ones[rows].sum(1)
            print(line + f" | same-index rows {np.round(row_ratios, 2).tolist()}")

    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
