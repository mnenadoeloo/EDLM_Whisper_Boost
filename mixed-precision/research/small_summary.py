"""Verify and summarize the small map experiment and the paired real-time replays."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from analyze import load_results, quantile, score, write_csv
from benchmark import audio_context, read_jsonl, sha256

MAPS = ["q4_0", "dec_cross", "m3_enc1"]


def summarize(out):
    stats, _ = load_results(out / "raw", "m3_panel", [m + "+ctx768" for m in MAPS])
    file_rows = []
    for lang in ("ru", "en"):
        for model in MAPS:
            s = stats[model + "+ctx768", lang]
            if s["clips"] != 32 or s["repeats"] != 1:
                raise ValueError("Expected one file run for each of the 32 clips per language")
            file_rows.append({"language": lang, "model": model, **s, "wall_s": s["rtf"] * s["audio_s"]})
    write_csv(out / "files.csv", file_rows)
    archive_path = out / "stream-archive.json"
    archived = archive_path.is_file()
    if archived:
        archive = json.loads(archive_path.read_text())
        for name in ("stream-meta.json", "stream.jsonl"):
            if sha256(out / name) != archive["sha256"][name]:
                raise ValueError(f"Archived streaming measurement changed: {name}")
    meta = json.loads((out / "stream-meta.json").read_text())
    # Both runs used the same model and native runtime. Their recorded hashes
    # can be checked on another machine without rebuilding the original binary.
    file_run = json.loads((out / "raw/m3_panel-m3_enc1+ctx768.json").read_text())
    runtime = {"whisper-cli": file_run["cli_sha256"], "m3_enc1.bin": file_run["model_sha256"],
               **file_run["libraries"]}
    manifest_path = out / "stream-manifest.jsonl"
    for src, expected in meta["sha256"].items():
        path = Path(src)
        if path.name in runtime:
            observed = runtime[path.name]
        elif archived and path.parts[:1] == ("demo",):
            # The sealed metadata identifies the measured code at source_commit.
            # A later checkout can rescore the trace without the old source tree.
            continue
        else:
            observed = sha256(manifest_path if path.name == manifest_path.name else path)
        if observed != expected:
            raise ValueError(f"Streaming experiment source changed: {src}")
    manifest = {r["id"]: r for r in read_jsonl(manifest_path)}
    panel = read_jsonl(Path("data/m3_panel.jsonl"))
    expected = [r for lang in ("ru", "en") for r in [s for s in panel if s["language"] == lang][:8]]
    if list(manifest.values()) != expected:
        raise ValueError("Streaming recordings differ from the planned subset")
    traces = read_jsonl(out / "stream.jsonl")
    if Counter((r["id"], r["backend"]) for r in traces) != Counter(
            (key, arm) for key in manifest for arm in ("cli", "resident")):
        raise ValueError("Incomplete or duplicated stream pair")
    pairs = []
    for key, sample in manifest.items():
        by_arm = {r["backend"]: r for r in traces if r["id"] == key}
        for r in by_arm.values():
            if any(r[k] != sample[k] for k in ("reference", "language", "wav_sha256", "duration_s")):
                raise ValueError(f"Input mismatch: {key}")
            if (r["model"], r["mode"], r["threads"]) != ("m3_enc1", "conservative", 4):
                raise ValueError(f"Incorrect runtime configuration: {key}")
            for u in r["updates"]:
                if not u["pcm_sha256"] or u["context"] != audio_context(u["window_s"], 768):
                    raise ValueError(f"Stream input or context mismatch: {key}")
            for metric, fields in (("first_text_s", ("partial", "stable")), ("first_confirmed_s", ("stable",))):
                first = next((u["time_s"] for u in r["updates"] if any(u[f] for f in fields)), None)
                if first != r[metric]:
                    raise ValueError(f"Invalid latency: {key}/{metric}")
        a, b = by_arm["cli"], by_arm["resident"]
        def inputs(trace):
            return [(u["pcm_sha256"], u["context"], u["final"]) for u in trace["updates"]]
        first_delta = None
        if a["first_text_s"] is not None and b["first_text_s"] is not None:
            first_delta = b["first_text_s"] - a["first_text_s"]
        pairs.append({"id": key, "language": sample["language"], "same_window_schedule": inputs(a) == inputs(b),
                      "cli_windows": len(a["updates"]), "resident_windows": len(b["updates"]),
                      "same_window_texts": [u["text"] for u in a["updates"]] == [u["text"] for u in b["updates"]],
                      "first_text_delta_s": first_delta,
                      "same_final_text": a["text"] == b["text"]})
    write_csv(out / "stream-pairs.csv", pairs)
    streams = []
    for lang in ("ru", "en"):
        for backend in ("cli", "resident"):
            rows = [r for r in traces if (r["language"], r["backend"]) == (lang, backend)]
            if Counter(r["pair_position"] for r in rows) != {1: 4, 2: 4}:
                raise ValueError("Unbalanced stream replay order")
            counts = [score({"reference": r["reference"], "prediction": r["text"]}) for r in rows]
            words, errors = (sum(r[k] for r in counts) for k in ("ref_words", "word_errors"))
            audio = sum(r["duration_s"] for r in rows)
            compute = sum(u["compute_s"] for r in rows for u in r["updates"])
            s = {"language": lang, "backend": backend, "clips": len(rows), "audio_s": audio,
                 "windows": sum(len(r["updates"]) for r in rows), "word_errors": errors, "words": words,
                 "wer": 100 * errors / words, "window_compute_s": compute, "compute_rtf": compute / audio,
                 "elapsed_s": sum(r["final_s"] for r in rows),
                 "first_text_before_audio_end": sum(r["first_text_s"] is not None and r["first_text_s"] < r["duration_s"]
                                                    for r in rows)}
            for name in ("first_text_s", "first_confirmed_s"):
                times = [r[name] for r in rows if r[name] is not None]
                s[name + "_observed"] = len(times)
                s[name + "_p50"] = statistics.median(times) if times else None
                s[name + "_p95"] = quantile(times, .95) if times else None
            streams.append(s)
    write_csv(out / "stream.csv", streams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/m3-small"))
    summarize(parser.parse_args().out)
