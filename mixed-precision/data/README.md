# Записи для замеров

В Git хранятся JSONL-манифесты с ID, эталонным текстом, длительностью, путём к WAV и SHA-256. Аудио восстанавливается в `.work/data/` в формате PCM16, моно, 16 кГц.

| Манифест | Golos, RU | LibriSpeech, EN | Для чего |
|---|---:|---:|---|
| [smoke](smoke.jsonl) | 1 | 1 | Проверка загрузки и прогрев |
| [calibration](calibration.jsonl) | 4 | 4 | Статистики на train |
| [dev](dev.jsonl) / [test](test.jsonl) | 8 / 24 | 8 / 24 | Подбор и отложенный тест пилота |
| [m3_dev](m3_dev.jsonl) | 32 | 32 | Выбор карты под устройство |
| [m3_test](m3_test.jsonl) | 1 891 | 948 | Общий тест |
| [m3_panel](m3_panel.jsonl) | 32 | 32 | Часть m3_test для повторных замеров времени |

**Golos** содержит короткие русские команды в условиях дальней записи. Использованы farfield validation и test. Идентификаторы дикторов отсутствуют. [Корпус и лицензия](https://www.openslr.org/114/) · [Ревизия данных](https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield/blob/c93949f7140beef4adc404e7b54841e957f81c54/README.md)

**LibriSpeech** содержит фрагменты чтения английских аудиокниг. Использованы чистые разделы train, dev и test по CC BY 4.0. Дикторы в пилоте, новом dev и новом test разделены. В расширенном test 16 дикторов. Оценки WER разных языков отражают также различие жанров и условий записи. [Корпус и лицензия](https://www.openslr.org/12/) · [Ревизия данных](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/clean)

Для малой выборки выполните первую команду. Для полного командного теста дополнительно выполните вторую. Нужны uv, curl и ffmpeg.

```sh
uv run --with pyarrow==25.0.1 python scripts/prepare_audio.py
uv run --with pyarrow==25.0.1 python scripts/prepare_m3_audio.py
```

Если WAV уже на месте, пилот проверяется командой `uv run scripts/prepare_audio.py`, расширенное аудио — командой `uv run scripts/prepare_m3_audio.py --verify-only`. В отчёте для расширенного сравнения использованы первые 128 записей каждого языка в зафиксированном случайном порядке. [ID этого среза](../results/m3/pilot-ids.json) · [Протокол](../results/m3/PROTOCOL.md)
