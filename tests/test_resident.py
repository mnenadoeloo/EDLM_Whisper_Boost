import json

import pytest

from demo.resident import Resident
from demo.server import CLI, MODELS, ROOT, transcribe


def test_resident_reuses_weights_with_independent_language_and_text_context():
    model = MODELS / "m3_enc1.bin"
    samples = [json.loads(line) for line in (ROOT / "data/smoke.jsonl").read_text().splitlines()]
    if not (model.is_file() and CLI.is_file() and all((ROOT / r["audio"]).is_file() for r in samples)):
        pytest.skip("Local M3 model, runtime and smoke audio are required")
    payloads = [(ROOT / r["audio"]).read_bytes() for r in samples]
    expected = [transcribe(wav, model, r["language"], 4, 768) for r, wav in zip(samples, payloads)]
    resident = Resident(model, ROOT / ".work/whisper.cpp")
    try:
        for i in (0, 1, 0):
            result = transcribe(payloads[i], model, samples[i]["language"], 4, 768, resident)
            assert result["text"] == expected[i]["text"]
            assert result["context"] == expected[i]["context"]
    finally:
        resident.close()
