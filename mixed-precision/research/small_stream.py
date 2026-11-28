"""Replay eight preselected clips per language through the CLI and resident demo backends."""
import argparse
import json
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from urllib.request import Request, urlopen

from benchmark import CLI, read_jsonl, sha256


def start(stack, resident):
    log = stack.enter_context(tempfile.TemporaryFile(mode="w+"))
    cmd = [sys.executable, "-m", "demo.server", "--port", "0", "--threads", "4"]
    if resident:
        cmd += ["--resident-model", "m3_enc1"]
    server = stack.enter_context(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True))
    stack.callback(server.terminate)
    url = server.stdout.readline().strip()
    if not url.startswith("http://127.0.0.1:"):
        log.seek(0)
        raise RuntimeError("Demo failed to start: " + log.read()[-2500:])
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".work/small-rerun"))
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    panel = Path("data/m3_panel.jsonl")
    samples = read_jsonl(panel)
    selected = [r for lang in ("ru", "en") for r in [s for s in samples if s["language"] == lang][:8]]
    manifest = out / "stream-manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    smoke = read_jsonl(Path("data/smoke.jsonl"))[0]
    with ExitStack() as stack:
        urls = [start(stack, resident) for resident in (False, True)]
        warmups = []
        for url in urls:
            request = Request(url + "/api/transcribe?model=m3_enc1&context=conservative&language=" + smoke["language"],
                              data=Path(smoke["audio"]).read_bytes(), headers={"Content-Type": "audio/wav"})
            with urlopen(request, timeout=60) as response:
                warmups.append(json.load(response))
        if warmups[0]["text"] != warmups[1]["text"]:
            raise ValueError("Warmup transcripts differ between backends")
        source = [CLI, Path(".work/models/m3_enc1.bin"), panel, manifest, Path("demo/resident.cpp"),
                  Path("demo/resident.py"), Path("demo/server.py"), Path("demo/replay.mjs"), Path("demo/stream.mjs")]
        source.extend(sorted(CLI.parents[1].glob("src/libwhisper.1.8.2.*")))
        meta = {"sha256": {str(p): sha256(p) for p in source}, "warmup_id": smoke["id"], "warmups": warmups,
                "model": "m3_enc1", "threads": 4, "floor": 768, "replay_rate": 1, "backend_order": "AB/BA per language"}
        (out / "stream-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        print(f"Both backends ready. Replaying {len(selected)} paired recordings.", flush=True)
        temporary = out / "stream.jsonl.part"
        with temporary.open("w") as log:
            subprocess.run(["node", "demo/replay.mjs", "m3_enc1", "conservative", manifest, *urls], stdout=log, check=True)
        temporary.replace(out / "stream.jsonl")


if __name__ == "__main__":
    main()
