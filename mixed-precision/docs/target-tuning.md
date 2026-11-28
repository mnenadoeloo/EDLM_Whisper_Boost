# Подбор конфигурации под CPU или GPU

На новом устройстве соберите whisper.cpp, затем измерьте кандидатов на одних и тех же записях. Скрипт ищет наименьший RTF при допустимом росте WER **отдельно на русском и английском**. Эталон — FP16 с полным окном на том же устройстве. RTF включает загрузку модели на каждый файл.

## Локальная машина

Для Apple GPU используйте `metal`, для NVIDIA — `cuda`, для процессора — `cpu`. Пример после [подготовки пилотного аудио](../data/README.md).

```sh
uv run scripts/setup_runtime.py --backend metal
uv run research/tune_target.py --backend metal --out .work/team-metal \
  --models fp16 q4_0 m3_enc1 --contexts full 768 --threads 4 \
  --budget-pp 0 0.5 1 2 --min-speedup 2 --repeats 2 --confirm test
uv run research/plot_target_curve.py .work/team-metal --output .work/team-metal/curve.png
```

- `--contexts full 768` сравнивает полное окно с адаптивным от 768 позиций. `--threads 4 2` дополнительно проверит два числа потоков.
- `dev.csv` содержит WER и RTF каждого варианта по языкам. `curve.csv` показывает, при каком бюджете меняется выбор. `--min-speedup 2` ищет минимальный бюджет для ускорения хотя бы вдвое.
- `selection.json` фиксирует выбор до теста. Результат на отложенной выборке попадает в `confirmation-test.json`. Для новой машины или сетки задайте новый `--out`. В старом каталоге можно пересчитать выбор по другому бюджету на тех же измерениях.

[Результаты CPU и Metal на M3 Pro](../results/targets/README.md) · [Подключение модели коллеги](team-merge.md)

## Colab CLI

Пример для частной рабочей копии и GPU T4. Сначала подготовьте WAV пилота локально. Упакуйте только код, карты и записи трёх выборок. Для командной модели дополнительно включите её `.bin` под `.work/models/`.

```sh
uv run python -c 'import json, pathlib; print("\n".join(json.loads(s)["audio"] for n in ("smoke","dev","test") for s in pathlib.Path(f"data/{n}.jsonl").read_text().splitlines()))' > .work/colab-wavs.txt
COPYFILE_DISABLE=1 tar --exclude=__pycache__ -czf .work/target-input.tgz -T .work/colab-wavs.txt \
  scripts research patches policies data/smoke.jsonl data/dev.jsonl data/test.jsonl
colab new -s whisper-tune --gpu T4
colab upload -s whisper-tune .work/target-input.tgz /content/target-input.tgz
colab exec -s whisper-tune --timeout 7200 <<'PY'
import pathlib, shutil, subprocess, sys
root = pathlib.Path("/content/whisper-tune")
root.mkdir(exist_ok=True)
def run(*cmd):
    subprocess.run(cmd, cwd=root, check=True)
run("tar", "-xzf", "/content/target-input.tgz")
run("nvidia-smi")
if not shutil.which("uv"):
    run(sys.executable, "-m", "pip", "install", "uv")
run("uv", "run", "--with", "cmake==4.1.0", "python", "scripts/setup_runtime.py", "--backend", "cuda", "--jobs", "2")
run("uv", "run", "research/tune_target.py", "--backend", "cuda", "--out", "results/t4",
    "--budget-pp", "0", "0.5", "1", "2", "--repeats", "1", "--confirm", "test")
run("tar", "-czf", "/content/whisper-t4-results.tgz", "results/t4")
PY
colab download -s whisper-tune /content/whisper-t4-results.tgz .work/whisper-t4-results.tgz
colab stop -s whisper-tune
```

В этой сборке для NVIDIA нужны драйвер с `nvidia-smi`, CUDA Toolkit с `nvcc` и CMake 3.24+. Подбор проверяет, что веса действительно размещены на GPU. В отчёте уже есть замеры CPU и Metal. CUDA предстоит измерить.
