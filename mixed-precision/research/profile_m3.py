"""Time shape-matched GGML operations on loaded weights in a counterbalanced order."""
import io
import csv
import subprocess
from pathlib import Path

from benchmark import BUILD
from m3_maps import OUT

ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = BUILD.resolve().parent
    lib = BUILD.resolve() / "ggml/src"
    binary = ROOT / ".work/m3-matmul"
    subprocess.run(["c++", "-O2", "-std=c++17", '-DWHISPER_VERSION="1.8.2"',
                    "-I", runtime / "src", "-I", runtime / "include", "-I", runtime / "ggml/include",
                    ROOT / "research/m3_matmul.cpp", "-L", lib, "-lggml", "-lggml-cpu", "-lggml-base",
                    f"-Wl,-rpath,{lib}", "-o", binary], check=True)
    formats = ["q4_0", "q5_0", "q8_0"]
    OUT.mkdir(exist_ok=True, parents=True)
    with (OUT / "matmul.csv").open("w") as dest:
        out = csv.writer(dest)
        out.writerow(["block", "format", "role", "K", "M", "N", "weight_buffer", "repeat", "us"])
        for i, block in enumerate([0, 5, 11]):
            for fmt in formats[i:] + formats[:i]:
                run = subprocess.run([binary, ROOT / f".work/models/{fmt}.bin", str(block)],
                                     capture_output=True, text=True)
                if run.returncode:
                    raise RuntimeError(f"{fmt}/{block} returned {run.returncode}: {run.stderr[-1800:]}")
                out.writerows([block, fmt, *row] for row in csv.reader(io.StringIO(run.stdout)))
                print(f"{fmt}, block {block}", flush=True)


if __name__ == "__main__":
    main()
