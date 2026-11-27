import argparse, csv, json, os, re, statistics, subprocess, sys, tempfile, threading, time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

def normalize_ru(s):
    s = s.lower().replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]|_", " ", s).split())

def parse_ms(log, key):
    m = re.search(rf"{key} time =\s*([\d.]+) ms", log)
    assert m, f"'{key} time' not found in whisper-cli log"
    return float(m.group(1))

def self_check():
    assert normalize_ru("Ёлка, 5 — раз-два!  ") == "елка 5 раз два"
    assert parse_ms("x:     load time =   12.50 ms\n", "load") == 12.5
    ballast = bytearray(200 * 2**20)
    ballast[::4096] = b"x" * len(ballast[::4096])
    _, rss, _ = run_cli([sys.executable, "-I", "-c", "pass"], gpu=False)
    assert rss < 100, f"RSS wrapper reports {rss:.0f} MB for an empty interpreter"
    del ballast

RSS_WRAPPER = ("import os, subprocess, sys; p = subprocess.Popen(sys.argv[2:]); _, s, u = os.wait4(p.pid, 0); "
               "open(sys.argv[1], 'w').write(str(u.ru_maxrss)); sys.exit(os.waitstatus_to_exitcode(s) & 255)")

def descendants(pid):
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        out.add(p)
        try:
            todo += [int(c) for c in Path(f"/proc/{p}/task/{p}/children").read_text().split()]
        except OSError:
            pass
    return out

def process_vram_mb(pids):
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    rows = [line.split(",") for line in r.stdout.strip().splitlines() if line.strip()]
    return sum(float(m) for p, m in rows if p.strip().isdigit() and int(p) in pids)

def run_cli(cmd, gpu):
    """-> (stderr log, peak RSS MB of the CLI, peak VRAM MB of the CLI or None)."""
    peak, stop = [0.0], threading.Event()
    with tempfile.TemporaryFile("w+") as err, tempfile.NamedTemporaryFile("r") as rss:
        proc = subprocess.Popen([sys.executable, "-I", "-c", RSS_WRAPPER, rss.name, *cmd],
                                stdout=subprocess.DEVNULL, stderr=err)

        def poll():
            while not stop.is_set():
                peak[0] = max(peak[0], process_vram_mb(descendants(proc.pid)))
                time.sleep(0.2)

        thread = threading.Thread(target=poll, daemon=True) if gpu else None
        if thread:
            thread.start()
        proc.wait()
        stop.set()
        if thread:
            thread.join()
        err.seek(0)
        log = err.read()
        if proc.returncode != 0:
            raise SystemExit(f"whisper-cli failed ({proc.returncode}):\n{log[-3000:]}")
        rss_mb = int(rss.read()) / 1024
    if gpu and peak[0] == 0:
        print("warning: whisper-cli not found in nvidia-smi compute apps, VRAM not recorded", flush=True)
    return log, rss_mb, (peak[0] or None) if gpu else None

def transcribe(a, model, rows):
    txts = [Path(r["path"] + ".txt") for r in rows]
    for t in txts:
        t.unlink(missing_ok=True)
    cmd = (["taskset", "-c", a.cpus] if a.cpus else []) + [
        a.cli, "-m", model, "-l", rows[0]["lang"], "-t", str(a.threads),
        "-bs", "1", "-bo", "1", "-nt", "-otxt"]
    if not a.fallback:
        cmd.append("-nf")
    if a.cpu:
        cmd.append("-ng")
    for r in rows:
        cmd += ["-f", r["path"]]
    log, rss, vram = run_cli(cmd, gpu=not a.cpu)
    missing = [t for t in txts if not t.exists()]
    assert not missing, f"no output for {len(missing)} files, e.g. {missing[0]}"
    hyps = [" ".join(t.read_text(encoding="utf-8", errors="replace").split()) for t in txts]
    for t in txts:
        t.unlink()
    seconds = (parse_ms(log, "total") - parse_ms(log, "load")) / 1000
    m = re.search(r"fallbacks =\s*(\d+) p /\s*(\d+) h", log)
    assert m, "'fallbacks' line not found in whisper-cli log"
    return hyps, seconds, rss, vram, int(m.group(1)), int(m.group(2))

def counts(refs, hyps, fn):
    out = []
    for r, h in zip(refs, hyps):
        m = fn(r, h)
        out.append((m.substitutions + m.deletions + m.insertions, m.hits + m.substitutions + m.deletions))
    return np.array(out, dtype=float)

def rate(c):
    return 100 * c[:, 0].sum() / c[:, 1].sum()

def bootstrap_diff(x, y, n=10000):
    """95% CI of WER(y) - WER(x), paired over utterances."""
    idx = np.random.default_rng(42).integers(0, len(x), (n, len(x)))
    d = 100 * (y[idx, 0].sum(1) / y[idx, 1].sum(1) - x[idx, 0].sum(1) / x[idx, 1].sum(1))
    return np.percentile(d, [2.5, 97.5])

def main():
    self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--hf", required=True, help="HF whisper-small dir (normalizer.json)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="pairs like B:C (model file stems)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="enable temperature fallback (no -nf)")
    ap.add_argument("--cpus", help="taskset core list, e.g. 0-3")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first N files of every subset")
    ap.add_argument("--repeats", type=int, default=1, help="RTF = median of N runs")
    a = ap.parse_args()

    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    normalize_en = EnglishTextNormalizer(json.loads((Path(a.hf) / "normalizer.json").read_text()))
    normalizers = {"en": normalize_en, "ru": normalize_ru}

    subsets = {}
    for manifest in a.manifests:
        for line in open(manifest, encoding="utf-8"):
            row = dict(zip(["path", "text", "id", "subset", "lang"], line.rstrip("\n").split("\t")))
            subsets.setdefault(row["subset"], []).append(row)

    names = [Path(m).stem for m in a.models]
    assert len(set(names)) == len(names), f"model file names must differ: {names}"
    pairs = [p.split(":") for p in a.compare]
    assert all(len(p) == 2 and set(p) <= set(names) for p in pairs), f"--compare must use {names}"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.txt").unlink(missing_ok=True)
    results, word_counts = [], {}

    for subset, rows in subsets.items():
        norm = normalizers[rows[0]["lang"]]
        rows = [r for r in rows if norm(r["text"])][: a.limit]
        duration = sum(sf.info(r["path"]).duration for r in rows)
        base_wer = None
        for model in a.models:
            name = Path(model).stem
            runs = [transcribe(a, model, rows) for _ in range(a.repeats)]
            hyps = runs[0][0]
            refs_n = [norm(r["text"]) for r in rows]
            hyps_n = [norm(h) for h in hyps]
            words = counts(refs_n, hyps_n, jiwer.process_words)
            chars = counts(refs_n, hyps_n, jiwer.process_characters)
            word_counts[subset, name] = words
            wer = rate(words)
            base_wer = wer if base_wer is None else base_wer
            vram = [r[3] for r in runs if r[3] is not None]
            results.append({
                "model": name, "subset": subset, "device": "cpu" if a.cpu else "gpu",
                "threads": a.threads, "cpus": a.cpus or "", "fallback": a.fallback, "files": len(rows),
                "fallbacks_p": runs[0][4], "fallbacks_h": runs[0][5],
                "audio_s": round(duration, 1), "wer": round(wer, 3), "cer": round(rate(chars), 3),
                "dwer_vs_baseline_pp": round(wer - base_wer, 3),
                "rtf": round(statistics.median(r[1] for r in runs) / duration, 4), "repeats": a.repeats,
                "peak_rss_mb": round(max(r[2] for r in runs), 1),
                "peak_vram_mb": max(vram) if vram else "",
                "model_mb": round(os.path.getsize(model) / 2**20, 1),
            })
            print(results[-1], flush=True)
            with open(out / f"pred_{subset}_{name}.tsv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["id", "ref", "hyp", "ref_norm", "hyp_norm"])
                w.writerows([r["id"], r["text"], h, rn, hn] for r, h, rn, hn in zip(rows, hyps, refs_n, hyps_n))
            with open(out / "results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=results[0].keys())
                w.writeheader()
                w.writerows(results)

        for x, y in pairs:
            lo, hi = bootstrap_diff(word_counts[subset, x], word_counts[subset, y])
            diff = rate(word_counts[subset, y]) - rate(word_counts[subset, x])
            line = f"{subset}: WER({y}) - WER({x}) = {diff:+.3f} pp, 95% CI [{lo:+.3f}, {hi:+.3f}]"
            print(line)
            with open(out / "compare.txt", "a") as f:
                f.write(line + "\n")

if __name__ == "__main__":
    main()
