"""Sequential, resumable Whisper CLI benchmark. The archived protocol defaults to CPU."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

BUILD = Path(".work/whisper.cpp/build")
CLI = BUILD / "bin/whisper-cli"
FLAGS = ["-ng", "-fa", "-bs", "1", "-bo", "1", "-tp", "0", "-nf", "-nt", "-otxt"]


def device_identity(backend):
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    else:
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith(("model name", "Hardware")) and ":" in line), platform.machine())
    identity = {"backend": backend, "cpu": cpu, "arch": platform.machine(), "platform": platform.platform(),
                "host": platform.node(), "logical_cpus": os.cpu_count()}
    if backend == "cuda":
        query = ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]
        identity["cuda_devices"] = subprocess.check_output(query, text=True).strip().splitlines()
        identity["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return identity


def backend_evidence(log, backend):
    if backend == "cpu":
        if re.search(r"use gpu\s*= 0", log) and re.search(r"backends\s*= 1\b", log) and "CPU :" in log:
            return {"backend": "cpu"}
    else:
        device = r"CUDA\d+" if backend == "cuda" else "Metal"
        active = re.search(rf"whisper_backend_init_gpu: using ({device}) backend", log)
        sizes = re.findall(rf"whisper_model_load:\s+({device}(?:_Private|_Shared)?) total size =\s*([\d.]+) MB", log)
        if (active and re.search(r"use gpu\s*= 1", log) and "failed to initialize" not in log
                and any(name.split("_")[0] == active[1] and float(size) > 0 for name, size in sizes)):
            if backend == "cuda":
                index = active[1].removeprefix("CUDA")
                hardware = re.search(rf"Device {index}: (.+?), compute capability", log)
            else:
                hardware = re.search(r"GPU name:\s*([^\n]+)", log)
            return {"backend": backend, "device": active[1], "description": hardware[1].strip() if hardware else active[1],
                    "weight_buffers_mb": {name: float(size) for name, size in sizes}}
    raise RuntimeError(f"Requested {backend} backend was not confirmed in whisper.cpp output:\n" + log[:4500])


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audio_context(duration, floor=256):
    return min(1500, max(floor, 64 * math.ceil((duration + 1.0) * 50 / 64)))


def config_floor(config):
    _, sep, mode = config.partition("+")
    if not sep:
        return 0
    if mode == "ctx":
        return 256
    match = re.fullmatch(r"ctx([1-9]\d{0,3})", mode)
    if not match or int(match[1]) > 1500:
        raise ValueError(f"Invalid audio context in {config}")
    return int(match[1])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def paired_order(samples):
    ordered = list(samples)
    random.Random(20261123).shuffle(ordered)
    return ordered


def m3_pilot(samples):
    ordered = paired_order(samples)
    return [r for lang in ("ru", "en") for r in [s for s in ordered if s["language"] == lang][:128]]


def read_samples(split):
    samples = read_jsonl(Path(f"data/{split}.jsonl"))
    if not samples or len({s["id"] for s in samples}) != len(samples):
        raise ValueError(f"Empty manifest or duplicate IDs in {split}")
    for sample in samples:
        audio = Path(sample["audio"])
        with wave.open(str(audio), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError(f"Expected 16 kHz mono PCM16: {audio}")
            duration = wav.getnframes() / wav.getframerate()
        if abs(duration - sample["duration_s"]) > 0.001:
            raise ValueError(f"Incorrect duration: {audio}")
        digest = sha256(audio)
        if sample.get("wav_sha256", digest) != digest:
            raise ValueError(f"Audio checksum mismatch: {audio}")
        sample.update(duration_s=duration, wav_sha256=digest)
    return samples


def run_clip(sample, model, floor, threads, directory, *, cli=CLI, backend="cpu"):
    ctx = audio_context(sample["duration_s"], floor) if floor else 0
    output = directory / "out"
    output.with_suffix(".txt").unlink(missing_ok=True)
    flags = FLAGS if backend == "cpu" else [flag for flag in FLAGS if flag != "-ng"]
    args = [str(cli), *flags, "-t", str(threads), "-ac", str(ctx), "-l", sample["language"],
            "-m", str(model), "-f", sample["audio"], "-of", str(output)]
    # wait4 reports this child alone. A cumulative RUSAGE_CHILDREN maximum
    # would incorrectly carry the peak of the FP16 process over to Q4.
    with tempfile.TemporaryFile() as stderr:
        start = time.perf_counter()
        with subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=stderr) as proc:
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        elapsed = time.perf_counter() - start
        stderr.seek(0)
        log = stderr.read().decode(errors="replace")
    if proc.returncode or not output.with_suffix(".txt").is_file():
        raise RuntimeError(f"{sample['id']} / {model}: {log[-3000:]}")
    evidence = backend_evidence(log, backend)
    stages = {name: float(ms) for name, ms in re.findall(
        r"whisper_print_timings:\s+(load|encode|decode|batchd|prompt) time\s*=\s*([\d.]+) ms", log)}
    return {"id": sample["id"], "language": sample["language"], "reference": sample["reference"],
            "duration_s": sample["duration_s"], "audio_ctx": ctx, "wall_s": elapsed,
            "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "stages_ms": stages, "prediction": output.with_suffix(".txt").read_text().strip(),
            **({"device_evidence": evidence} if backend != "cpu" else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["smoke", "dev", "test", "m3_dev", "m3_test", "m3_panel"])
    parser.add_argument("configs", nargs="+", help="model stem, optionally suffixed with +ctx or +ctx768")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"),
                        help="use the target build and record device identity")
    parser.add_argument("--output", type=Path, default=Path("results/raw"))
    parser.add_argument("--overrides", type=Path, help="look for run-specific models here before .work/models")
    parser.add_argument("--pilot", action="store_true", help="measure the first 128 paired test clips per language")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or len(set(args.configs)) != len(args.configs):
        parser.error("Expected positive repeats/threads and unique configs")
    samples = read_samples(args.split)
    if args.pilot and args.split != "m3_test":
        parser.error("--pilot applies to m3_test")
    measured = m3_pilot(samples) if args.pilot else samples
    warmup = read_samples("smoke")[0]
    args.output.mkdir(exist_ok=True, parents=True)
    backend = args.backend or "cpu"
    build = BUILD if backend == "cpu" else BUILD.with_name("build-" + backend)
    cli = build / "bin/whisper-cli"
    libraries = {p.resolve() for p in build.rglob("*") if p.is_file() and (".so" in p.suffixes or ".dylib" in p.suffixes)}
    spec = {"split": args.split, "threads": args.threads,
            "flags": FLAGS if backend == "cpu" else [f for f in FLAGS if f != "-ng"], "host": platform.node(),
            "platform": platform.platform(), "cli_sha256": sha256(cli),
            "libraries": {p.name: sha256(p) for p in sorted(libraries)},
            "audio_sha256": {s["id"]: s["wav_sha256"] for s in samples},
            "manifest_sha256": sha256(Path(f"data/{args.split}.jsonl")),
            "warmup": warmup["id"], "context_rule": "min(1500,max(256,64*ceil((duration_s+1)*50/64)))"}
    if args.backend:
        spec["hardware"] = device_identity(backend)
    paired = args.split in {"m3_test", "m3_panel"} and not args.backend
    if paired:
        if len(args.configs) != 2 and not (args.split == "m3_panel" and len(args.configs) == 3):
            parser.error("Use two configurations for confirmation or two/three for the timing panel")
        spec["order"] = "seeded clip order; adjacent model pairs, alternating order within language"
        if len(args.configs) == 3:
            spec["order"] = "seeded clip order; adjacent triplets, rotating order within language"
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        remaining = []
        for config in args.configs:
            stem = config.split("+", 1)[0]
            floor = config_floor(config)
            model = Path(".work/models") / (stem + ".bin")
            if args.overrides and (args.overrides / model.name).is_file():
                model = args.overrides / model.name
            setting = dict(spec, config=config, model_bytes=model.stat().st_size, model_sha256=sha256(model))
            if floor:
                setting["context_rule"] = setting["context_rule"].replace("max(256", f"max({floor}")
            base = args.output / f"{args.split}-{config}"
            meta, raw = base.with_suffix(".json"), base.with_suffix(".jsonl")
            if raw.exists() and not meta.exists():
                raise ValueError(f"Missing provenance for {raw}")
            if meta.exists() and json.loads(meta.read_text()) != setting:
                raise ValueError(f"Incompatible provenance for {raw}, choose a new output directory")
            meta.write_text(json.dumps(setting, indent=2, ensure_ascii=False) + "\n")
            seen = {(r["id"], r["repeat"]) for r in read_jsonl(raw)} if raw.exists() else set()
            pending = [(rep, s, config, model, floor, raw) for rep in range(1, args.repeats + 1)
                       for s in measured if (s["id"], rep) not in seen]
            if pending:
                print(f"warmup {config} ({len(pending)} measured calls pending)", flush=True)
                run_clip(warmup, model, floor, args.threads, directory, cli=cli, backend=backend)
            remaining.extend(pending)
        random.Random(20261121).shuffle(remaining)
        remaining.sort(key=lambda task: task[0])
        if paired:
            shuffled = paired_order(samples)
            clip_order = {row["id"]: i for i, row in enumerate(shuffled)}
            lang_order = {r["id"]: i for lang in ("en", "ru") for i, r in
                          enumerate(row for row in shuffled if row["language"] == lang)}
            remaining.sort(key=lambda t: (t[0], clip_order[t[1]["id"]],
                                          (lang_order[t[1]["id"]] + t[0] + args.configs.index(t[2])) % len(args.configs)))
        for i, (rep, sample, config, model, floor, raw) in enumerate(remaining, 1):
            row = run_clip(sample, model, floor, args.threads, directory, cli=cli, backend=backend)
            row.update(config=config, split=args.split, repeat=rep)
            with raw.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i == 1 or i % 20 == 0 or i == len(remaining):
                print(f"{i}/{len(remaining)} {config} {row['id']} {row['wall_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
