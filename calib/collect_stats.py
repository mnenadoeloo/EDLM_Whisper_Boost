import argparse, json, os
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

CONV = {
    "self_attn.k_proj": "attn.key", "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value", "self_attn.out_proj": "attn.out",
    "encoder_attn.q_proj": "cross_attn.query", "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out", "encoder_attn.k_proj": "cross_attn.key",
    "fc1": "mlp.0", "fc2": "mlp.2",
}

def ggml_name(hf_name):
    p = hf_name.split(".")
    if p[0] == "model":
        p = p[1:]
    if len(p) not in (4, 5) or p[1] != "layers" or ".".join(p[3:]) not in CONV:
        return None
    return f"{p[0]}.blocks.{p[2]}.{CONV['.'.join(p[3:])]}.weight"

def takes_encoder_frames(name):
    return name.startswith("encoder.") or ".cross_attn.key." in name or ".cross_attn.value." in name

def self_check():
    assert ggml_name("model.decoder.layers.2.encoder_attn.k_proj") == "decoder.blocks.2.cross_attn.key.weight"
    assert ggml_name("model.decoder.layers.2.fc1") == "decoder.blocks.2.mlp.0.weight"
    assert ggml_name("proj_out") is None
    assert takes_encoder_frames("encoder.blocks.0.mlp.0.weight")
    assert takes_encoder_frames("decoder.blocks.2.cross_attn.value.weight")
    assert not takes_encoder_frames("decoder.blocks.2.cross_attn.query.weight")
    assert not takes_encoder_frames("decoder.blocks.2.attn.key.weight")

def load_wav(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    return x[:480000]

def main():
    self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--audio-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default=None, help="for lines without a lang column; default = auto-detect")
    ap.add_argument("--no-decoder", action="store_true", help="encoder layers only")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model).to(a.device).eval()

    sums, counts = {}, {}
    state = {"enc_frames": 1500}

    def hook(name):
        enc = takes_encoder_frames(name)
        def f(mod, inp):
            x = inp[0].detach().float()
            if enc:  # drop the zero-padding up to 30 s
                assert x.shape[0] == 1 and x.shape[1] == 1500, f"{name}: unexpected input {tuple(x.shape)}"
                x = x[:, : state["enc_frames"]]
            x = x.reshape(-1, x.shape[-1])
            sums[name] = sums.get(name, 0) + (x.double() ** 2).sum(0).cpu()
            counts[name] = counts.get(name, 0) + x.shape[0]
        return f

    for n, m in model.named_modules():
        g = ggml_name(n) if isinstance(m, torch.nn.Linear) else None
        if g and not (a.no_decoder and g.startswith("decoder.")):
            m.register_forward_pre_hook(hook(g))

    paths = [l.strip() for l in open(a.audio_list) if l.strip()]
    with torch.no_grad():
        for i, line in enumerate(paths):
            p, _, lang = line.partition("\t")
            lang = lang or a.language
            wav = load_wav(p)
            # mel frames = samples/160, encoder frames = mel/2
            state["enc_frames"] = max(1, len(wav) // 160 // 2)
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(a.device)
            kw = {"language": lang, "task": "transcribe"} if lang else {}
            if a.no_decoder:
                model.model.encoder(feats)
            else:
                model.generate(feats, max_new_tokens=a.max_new_tokens, **kw)
            print(f"[{i+1}/{len(paths)}] {lang} {p}", flush=True)
            if i == 0:
                enc_n = counts["encoder.blocks.0.attn.query.weight"]
                bad = {k: v for k, v in counts.items() if takes_encoder_frames(k) and v != enc_n}
                assert not bad, f"expected {enc_n} encoder frames, got {bad}"

    out = {k: (sums[k] / counts[k]).float().numpy() for k in sums}
    out = {k: v / max(float(v.mean()), 1e-12) for k, v in out.items()}
    np.savez(a.out + ".npz", **out)
    json.dump({"model": a.model, "audio": [os.path.basename(x) for x in paths], "language": a.language,
               "decoder": not a.no_decoder, "normalisation": "E[x^2] / mean_j E[x^2]",
               "padding": "encoder frames beyond real audio length dropped (encoder and cross_attn key/value); silence kept",
               "tokens": counts}, open(a.out + ".json", "w"), indent=1)
    print(f"saved {len(out)} tensors")

if __name__ == "__main__":
    main()
