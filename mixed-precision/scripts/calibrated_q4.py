"""Search representable Q4_0 scales with a diagonal activation second-moment objective."""
import hashlib
import numpy as np


def pack_q4(data, reference, channel_weights):
    # Rows follow ggml ne[0]; Q4_0 stores a little-endian fp16 scale and 16 packed bytes.
    weights = np.asarray(channel_weights, dtype=np.float64)
    values = np.frombuffer(data, dtype="<f2").astype(np.float32).reshape(-1, len(weights))
    values = values.reshape(-1, 32)
    importance = np.broadcast_to(weights, (values.size // len(weights), len(weights))).reshape(-1, 32)
    reference = np.frombuffer(reference, dtype=np.uint8).reshape(-1, 18)
    base = reference[:, :2].copy().view("<f2").astype(np.float32)
    nibbles = reference[:, 2:]
    codes = np.concatenate([nibbles & 15, nibbles >> 4], axis=1).astype(np.int8) - 8
    best_scales, best_codes = base.copy(), codes.copy()

    def loss(scales, quantized):
        return ((values - scales * quantized) ** 2 * importance).sum(axis=1)

    best_loss = loss(base, codes)
    initial = best_loss.sum()
    for multiplier in np.linspace(0.7, 1.2, 21):
        # Score after fp16 scale rounding, as the inference kernel sees this precision.
        scales = (base * float(multiplier)).astype("<f2").astype(np.float32)
        new_codes = np.clip(np.rint(values / np.where(scales == 0, 1, scales)), -8, 7).astype(np.int8)
        error = loss(scales, new_codes)
        better = error < best_loss
        best_scales[better] = scales[better]
        best_codes[better] = new_codes[better]
        best_loss[better] = error[better]
    codes = (best_codes + 8).astype(np.uint8)
    packed = np.empty_like(reference)
    packed[:, :2] = best_scales.astype("<f2").view("u1")
    packed[:, 2:] = codes[:, :16] | (codes[:, 16:] << 4)
    diagnostics = {
        "rtn_weighted_sse": float(initial), "calibrated_weighted_sse": float(best_loss.sum()),
        "blocks": len(base), "changed_blocks": int(np.any(reference != packed, axis=1).sum()),
        "rtn_payload_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "calibrated_payload_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
    }
    return packed.tobytes(), diagnostics
