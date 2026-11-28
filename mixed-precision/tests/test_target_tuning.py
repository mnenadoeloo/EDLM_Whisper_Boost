import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from benchmark import audio_context, backend_evidence, config_floor
from tune_target import best, best_for_speed, frontier, relative


def test_quality_budget_applies_to_each_language_and_frontier_changes_only_at_breakpoints():
    def row(cfg, t, rtf, ru, en):
        return dict(config=cfg, threads=t, rtf=rtf, wer_ru=ru, wer_en=en, model_mib=100)
    rows = [row("fp16", 4, .8, 20, 5), row("mixed", 4, .4, 19, 5), row("mixed", 2, .5, 19, 5),
            row("q4", 4, .25, 18, 6), row("team", 4, .12, 28, 1)]
    relative(rows, 4)
    assert best(rows, 0)["config"] == "mixed"
    assert best(rows, .99)["config"] == "mixed"
    assert best(rows, 1)["config"] == "q4"
    assert [(p["budget_pp"], p["config"]) for p in frontier(rows)] == [(0, "mixed"), (1, "q4"), (8, "team")]
    assert best(rows, 0)["speedup"] == 2
    assert best_for_speed(frontier(rows), 2)["config"] == "mixed"
    assert best_for_speed(frontier(rows), 3)["budget_pp"] == 1
    assert best_for_speed(frontier(rows), 10) is None


def test_repeat_quality_includes_changed_predictions_without_changing_archived_scoring():
    from analyze import aggregate, score
    a = {"id": "ru", "repeat": 1, "reference": "два слова", "prediction": "два слова", "wall_s": .2,
         "duration_s": 1., "peak_rss_bytes": 10, "stages_ms": {"encode": 1, "load": 1, "decode": 1}}
    b = {**a, "repeat": 2, "prediction": "два"}
    old = {"ru": score(a)}
    pooled = {(r["id"], r["prediction"]): score(r) for r in (a, b)}
    assert aggregate([a, b], 10, old)["wer"] == 0
    assert aggregate([a, b], 10, old, pooled)["wer"] == 25


@pytest.mark.parametrize(("backend", "device"), [("cuda", "CUDA0"), ("metal", "Metal")])
def test_backend_requires_selected_gpu_and_positive_weights(backend, device):
    log = f"use gpu = 1\nwhisper_backend_init_gpu: using {device} backend\n"
    def allocation(buffer, size):
        return log + f"whisper_model_load: {buffer} total size = {size} MB\n"
    evidence = backend_evidence(allocation(device, 159.1), backend)
    assert evidence["device"] == device and evidence["weight_buffers_mb"] == {device: 159.1}
    for wrong in (allocation("CPU", 159), allocation(device, 0), allocation("CUDA1", 4),
                  allocation(device, 159) + f"failed to initialize {device} backend"):
        with pytest.raises(RuntimeError):
            backend_evidence(wrong, backend)


def test_contexts_support_custom_floor_and_archived_names():
    assert [config_floor(s) for s in ("fp16", "w+ctx", "w+ctx512", "w+ctx768")] == [0, 256, 512, 768]
    assert audio_context(2, config_floor("w+ctx512")) == 512
    with pytest.raises(ValueError):
        config_floor("w+ctx1501")
