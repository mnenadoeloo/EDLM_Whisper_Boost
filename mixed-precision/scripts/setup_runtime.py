"""Build the pinned whisper.cpp for CPU, Metal or CUDA and fetch multilingual Small."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from export_model import ROOT, SOURCE, SOURCE_SHA, sha256

UPSTREAM = "https://github.com/ggml-org/whisper.cpp"
COMMIT = "4979e04f5dcaccb36057e059bbaed8a2f5288315"
HF_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
RUNTIME = ROOT / ".work/whisper.cpp"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--jobs", type=int, help="parallel compiler jobs (CUDA default: 2)")
    args = parser.parse_args()
    backend = args.backend
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    if not RUNTIME.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v1.8.2", UPSTREAM, RUNTIME], check=True)
    git = ["git", "-C", RUNTIME]
    if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit(f"Expected whisper.cpp at {COMMIT} in {RUNTIME}")
    patch = ROOT / "patches/whisper-mixed.patch"
    if subprocess.run([*git, "apply", "--reverse", "--check", patch], capture_output=True).returncode:
        subprocess.run([*git, "apply", "--check", patch], check=True)
        subprocess.run([*git, "apply", patch], check=True)
    cmake = [os.environ.get("CMAKE") or shutil.which("cmake") or ""]
    if not cmake[0]:
        cmake = ["uv", "run", "--no-project", "--with", "cmake==4.1.0", "cmake"]
    build = RUNTIME / ("build" if backend == "cpu" else f"build-{backend}")
    gpu_flags = [f"-DGGML_METAL={'ON' if backend == 'metal' else 'OFF'}",
                 f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}"]
    if backend == "cuda":
        gpu_flags.append("-DCMAKE_CUDA_ARCHITECTURES=native")
    subprocess.run([*cmake, "-S", RUNTIME, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                    *gpu_flags, "-DWHISPER_COREML=OFF", "-DGGML_BLAS=OFF",
                    "-DWHISPER_BUILD_TESTS=OFF", "-DBUILD_SHARED_LIBS=ON"], check=True)
    targets = ["whisper-cli", "quantize"]
    if backend != "cpu":
        targets.insert(0, f"ggml-{backend}")
    subprocess.run([*cmake, "--build", build, "--parallel", str(args.jobs or (2 if backend == "cuda" else 8)),
                    "--target", *targets], check=True)
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE.exists():
        if sha256(SOURCE) != SOURCE_SHA:
            raise SystemExit(f"Existing {SOURCE} has a different checksum")
        return
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{HF_REVISION}/ggml-small.bin"
    with tempfile.NamedTemporaryFile(dir=SOURCE.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", temp, url], check=True)
        if sha256(temp) != SOURCE_SHA:
            raise ValueError("Downloaded GGML model has a different SHA-256")
        os.replace(temp, SOURCE)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
