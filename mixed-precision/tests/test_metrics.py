import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from analyze import distance, normalize, quantile, score, select
from benchmark import audio_context


def test_normalization_and_corpus_counts():
    assert normalize(" Ёжик, чёрно-белый! It’s MINE. ") == "ежик черно белый it's mine"
    assert score({"reference": "a b c", "prediction": "a x c d"}) == {
        "word_errors": 2, "ref_words": 3, "char_errors": 2, "ref_chars": 3}
    assert distance(["a", "b"], []) == 2
    assert distance([], ["a"]) == 1
    assert quantile([5, 1, 3], .5) == 3


def test_worst_language_budget_and_context_coverage():
    data = {(k, l): {"wer": wer, "rtf": rtf, "model_mib": 1}
            for k, en, ru, rtf in [("fp16", 10, 20, .5), ("fast", 8, 22, .2), ("ok", 11, 19, .3)]
            for l, wer in [("en", en), ("ru", ru), ("all", 0)]}
    assert select(data, ["fast", "ok", "fp16"])["config"] == "ok"
    assert not select(data, ["fast"])["meets_dev_budget"]
    assert [audio_context(x) for x in [0.3, 8.0, 30, 60]] == [256, 512, 1500, 1500]


def test_paired_uncertainty_uses_words_and_resamples_utterances_together():
    from analyze import paired_wer
    scores = {"parent": {"a": {"word_errors": 0, "ref_words": 2}, "b": {"word_errors": 1, "ref_words": 4}},
              "child": {"a": {"word_errors": 1, "ref_words": 2}, "b": {"word_errors": 3, "ref_words": 4}}}
    assert paired_wer(scores, "child", "parent", ["a", "b"]) == (50.0, 50.0)


def test_paired_speed_uses_same_utterances_in_both_arms():
    from analyze import paired_speed
    times = {"base": {"a": 8, "b": 4, "c": 2}, "fast": {"a": 4, "b": 2, "c": 1}}
    assert paired_speed(times, "fast", "base", [["a", "b"], ["c"]]) == (2.0, 2.0)


def test_m3_cluster_resampling_preserves_matching_errors_words_and_times():
    from m3_summary import paired_intervals
    # Two clusters of different sizes have the same WER difference and wall-time ratio.
    intervals = paired_intervals([(1, 50, 3.0, 2.0), (3, 150, 12.0, 8.0)])
    assert intervals == {"delta_wer_low_pp": 2., "delta_wer_high_pp": 2.,
                         "speedup_low": 1.5, "speedup_high": 1.5}


def test_m3_pilot_is_the_same_language_balanced_prefix_as_the_full_run():
    from benchmark import m3_pilot, paired_order
    rows = [{"id": i, "language": "ru" if i % 3 else "en"} for i in range(600)]
    order = paired_order(rows)
    chosen = m3_pilot(rows)
    assert len({r["id"] for r in chosen}) == 256
    for lang in ("ru", "en"):
        assert [r for r in chosen if r["language"] == lang] == [r for r in order if r["language"] == lang][:128]
