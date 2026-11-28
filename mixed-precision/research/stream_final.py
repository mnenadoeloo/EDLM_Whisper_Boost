"""Compare whole-file and streaming recognition, with one warmed model and timed PCM input."""
import argparse
import json
import statistics
import subprocess
import tempfile
import wave
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from analyze import score, write_csv
from benchmark import audio_context, device_identity, read_jsonl, sha256
from small_stream import start

POLICIES = ("batch", "classic", "endpoint")
MANIFEST = Path("results/m3-small/stream-manifest.jsonl")


def conversation(clips, path):
    # Keep each original recording intact; insert one second between recordings only.
    with wave.open(str(path), "wb") as dst:
        dst.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        for i, clip in enumerate(clips):
            wav = Path(clip["audio"])
            if sha256(wav) != clip["wav_sha256"]:
                raise ValueError(f"Recording changed: {wav}")
            with wave.open(str(wav), "rb") as src:
                if i:
                    dst.writeframes(bytes(32000))
                dst.writeframes(src.readframes(src.getnframes()))
    with wave.open(str(path), "rb") as result:
        duration = result.getnframes() / result.getframerate()
    return {"id": path.stem, "language": clips[0]["language"], "components": [c["id"] for c in clips],
            "wav_sha256": sha256(path), "duration_s": duration,
            "reference": " ".join(c["reference"] for c in clips)}


def measure(out):
    model = Path(".work/models/m3_enc1.bin")
    source = [Path(s) for s in ("demo/server.py", "demo/resident.cpp", "demo/resident.py", "demo/stream.mjs",
                                "demo/endpoint.mjs", "research/stream_final.mjs", "research/stream_final.py")]
    libs = list(Path(".work/whisper.cpp/build/src").glob("libwhisper.1.8.2.*"))
    meta = {"device": device_identity("cpu"), "model": model.stem, "model_sha256": sha256(model),
            "threads": 4, "context_floor": 768, "playback_rate": 1, "manifest_sha256": sha256(MANIFEST),
            "sources": {str(p): sha256(p) for p in [*source, *libs]}}
    with ExitStack() as stack:
        scratch = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        clips = read_jsonl(MANIFEST)
        meta["sessions"] = []
        inputs = list(clips)
        for language in ("ru", "en"):
            path = scratch / f"session-{language}.wav"
            session = conversation([r for r in clips if r["language"] == language][:4], path)
            meta["sessions"].append(session)
            inputs.append({**session, "audio": str(path)})
        schedule = scratch / "inputs.jsonl"
        schedule.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs))
        url = start(stack, resident=True)
        tmp = out / "runs.jsonl.part"
        with tmp.open("w") as dest:
            subprocess.run(["node", "research/stream_final.mjs", url, schedule], stdout=dest, check=True)
    tmp.replace(out / "runs.jsonl")
    (out / "profile.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def summarize(out):
    manifest = {r["id"]: r for r in read_jsonl(MANIFEST)}
    profile = json.loads((out / "profile.json").read_text())
    if profile["manifest_sha256"] != sha256(MANIFEST):
        raise ValueError("Frozen recording manifest changed")
    for s in profile["sessions"]:
        clips = [r for r in manifest.values() if r["language"] == s["language"]][:4]
        if (s["components"] != [r["id"] for r in clips] or
                s["reference"] != " ".join(r["reference"] for r in clips) or
                abs(s["duration_s"] - sum(r["duration_s"] for r in clips) - 3.0) > 1e-6):
            raise ValueError(f"Composite recording differs from the fixed recipe: {s['id']}")
        manifest[s["id"]] = s
    rows = read_jsonl(out / "runs.jsonl")
    if Counter((r["id"], r["policy"]) for r in rows) != Counter((k, p) for k in manifest for p in POLICIES):
        raise ValueError("Expected one matched triplet per recording")
    for r in rows:
        sample = manifest[r["id"]]
        if any(sample[k] != r[k] for k in ("wav_sha256", "language", "duration_s")):
            raise ValueError(f"Input mismatch: {r['id']}")
        if not (abs(r["capture_s"] - r["duration_s"]) < .15 and r["final_s"] >= r["capture_s"]):
            raise ValueError(f"Replay clock mismatch: {r['id']}")
        for call in r["calls"]:
            if (call["audio_s"] > call["available_s"] + 1e-6 or call["end_s"] > r["final_s"] or
                    call["context"] != audio_context(call["audio_s"], 768)):
                raise ValueError(f"Invalid capture timing or audio context: {r['id']}")
        if r["policy"] == "batch" and (len(r["calls"]) != 1 or r["calls"][0]["start_s"] < r["capture_s"]):
            raise ValueError(f"Batch processed audio before capture ended: {r['id']}")
        r.update(score({"reference": sample["reference"], "prediction": r["text"]}))

    summary = []
    groups = {}
    for r in rows:
        workload = "four_utterances" if "components" in manifest[r["id"]] else "single_clip"
        groups.setdefault((workload, r["language"]), []).append(r)
    for (workload, lang), runs in groups.items():
        batch = {r["id"]: r for r in runs if r["policy"] == "batch"}
        classic = {r["id"]: r for r in runs if r["policy"] == "classic"}
        for policy in POLICIES:
            group = [r for r in runs if r["policy"] == policy]
            compute = sum(c["compute_s"] for r in group for c in r["calls"])
            errors, words = (sum(r[k] for r in group) for k in ("word_errors", "ref_words"))
            diff = [batch[r["id"]]["tail_s"] - r["tail_s"] for r in group]
            vs_classic = [classic[r["id"]]["tail_s"] - r["tail_s"] for r in group]
            first = [r["first_s"] for r in group if r["first_s"] is not None]
            summary.append({"workload": workload, "language": lang, "policy": policy, "clips": len(group), "word_errors": errors,
                            "words": words, "wer": 100 * errors / words, "first_observed": len(first),
                            "first_p50_s": statistics.median(first) if first else None,
                            "tail_p50_s": statistics.median(r["tail_s"] for r in group),
                            "final_from_start_p50_s": statistics.median(r["final_s"] for r in group),
                            "compute_rtf": compute / sum(r["duration_s"] for r in group),
                            "paired_tail_saved_p50_s": statistics.median(diff),
                            "faster_tail_than_batch": sum(d > 0 for d in diff),
                            "paired_tail_saved_vs_classic_p50_s": statistics.median(vs_classic),
                            "faster_tail_than_classic": sum(d > 0 for d in vs_classic),
                            "first_before_end": sum(r["first_s"] is not None and r["first_s"] < r["capture_s"]
                                                    for r in group)})
    write_csv(out / "summary.csv", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".work/stream-final"))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        measure(args.out)
    summarize(args.out)
