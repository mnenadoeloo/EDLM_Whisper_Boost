# Как повторить эксперимент

Команды выполняются из корня проекта после [сборки CPU и подготовки пилотного аудио](../README.md#для-команды). Сохранённые результаты лежат в `results/`. Для новых запусков используйте `.work/`.

## Вес модели и аудиоокно

```sh
for model in q4_0 q5_0 q8_0 enc_attn enc_ffn dec_self dec_cross dec_ffn encoder; do
  uv run scripts/export_model.py "policies/$model.json" ".work/models/$model.bin"
done
uv run research/experiment.py dev
uv run research/experiment.py test
uv run research/figures.py .work/rerun
```

На dev сравниваются пять замен Q4 на Q8, контрольные модели и варианты окна. Из моделей, укладывающихся в +1 п.п. WER на каждом языке, выбирается самая быстрая. На отложенном test выполняются три повтора. В `.work/rerun` появятся выбранная карта, таблицы и графики. [Чувствительность групп](../results/sensitivity.png) · [Сравнение на test](../results/test-comparison.png)

Для WER суммируем словесные ошибки по корпусу и делим на число эталонных слов. Регистр и пунктуация снимаются по общим правилам, «ё» и «е» приравниваются. RTF включает запуск CLI и загрузку весов. [Расчёт метрик](../research/analyze.py)

## Калибровка трёх матриц

Для сравнения обычного округления, подбора масштаба и подбора с учётом активаций экспортируйте два дополнительных варианта. Они используют замороженные [статистики train](../results/models/calibration.json). Чтобы собрать статистики заново, выполните `uv run scripts/collect_calibration.py`.

```sh
for model in dec_cross_grid dec_cross_cal; do
  uv run --python 3.12 --with numpy==2.3.3 scripts/export_model.py "policies/$model.json" ".work/models/$model.bin"
done
uv run research/experiment.py dev --with-calibration --out .work/rerun-cal
uv run research/experiment.py test --with-calibration --out .work/rerun-cal
```

## Карта по скорости операций CPU

Нужны модели из первого раздела и [расширенное аудио](../data/README.md). Следующие команды измеряют стоимость операций, строят карты и выбирают одну на dev. По умолчанию тест ограничен 128 записями на язык. Для полного командного замера добавьте `--full` к двум последним командам.

```sh
export WHISPER_M3_RESULTS=.work/m3-rerun
uv run research/profile_m3.py
uv run research/m3_maps.py
uv run research/m3_experiment.py
uv run research/m3_summary.py
```

Чтобы пересчитать опубликованные таблицы по исходным транскрипциям без распознавания, выполните `env -u WHISPER_M3_RESULTS uv run research/m3_summary.py`. Разбиение, порядок прогонов и метод подсчёта интервалов закреплены в [протоколе](../results/m3/PROTOCOL.md).

<details>
<summary>Продолжить исходный замер на том же M3 Pro</summary>

В журнале есть 668 полных пар, из них 256 попали в отчёт. На исходном устройстве с той же сборкой можно продолжить в копии журнала. Для первой команды используйте новый каталог. Бенчмарк сверяет параметры и хеши и пропускает готовые записи.

```sh
export WHISPER_M3_RESULTS=.work/m3-continue
mkdir "$WHISPER_M3_RESULTS"
cp -R results/m3/raw results/m3/selection.json "$WHISPER_M3_RESULTS/"
uv run research/benchmark.py m3_test enc_attn+ctx768 m3_enc1+ctx768 --repeats 1 --output "$WHISPER_M3_RESULTS/raw"
uv run research/benchmark.py m3_panel enc_attn+ctx768 m3_enc1+ctx768 --repeats 3 --output "$WHISPER_M3_RESULTS/raw"
uv run research/m3_summary.py --full
```

</details>

## Три карты и стриминг

Нужны CPU-сборка, Node.js, расширенная выборка и три веса в `.work/models/`. Скрипт стриминга сам запускает два локальных сервера и подаёт одинаковый звук в реальном времени.

```sh
for model in q4_0 dec_cross m3_enc1; do
  uv run scripts/export_model.py "policies/$model.json" ".work/models/$model.bin"
done
uv run research/benchmark.py m3_panel q4_0+ctx768 dec_cross+ctx768 m3_enc1+ctx768 --repeats 1 --output .work/small-rerun/raw
uv run research/small_stream.py --out .work/small-rerun
uv run research/small_summary.py --out .work/small-rerun
```

Для пересчёта первого этапа достаточно `uv run research/small_summary.py`. Историческая трасса и версия программы зафиксированы [контрольными суммами](../results/m3-small/stream-archive.json). [Трассы потока](../results/m3-small/stream.jsonl) · [Парные результаты](../results/m3-small/stream-pairs.csv)

**Второй этап** сравнивает файл целиком, исходный поток и подготовку финала во время паузы. Нужны те же WAV, одна модель `m3_enc1` и четыре потока CPU. Один сервер держит веса в памяти для всех трёх режимов. Для короткого повторения на своём устройстве выполните первую команду. Вторая пересчитывает сохранённый опыт без аудио и модели.

```sh
uv run research/stream_final.py --out .work/stream-final
uv run research/stream_final.py --out results/stream-final --summarize-only
```

В `summary.csv` время первого текста отсчитывается от начала подачи аудио, задержка финала — от последнего блока PCM. Корпусный WER и время работы декодера на секунду аудио (RTF) посчитаны отдельно для RU и EN.

## Быстрые проверки после изменений

```sh
uv run --python 3.12 --with pytest --with numpy==2.3.3 python -m pytest -q
node --test demo/stream.test.mjs
```

Для проверок с реальным загрузчиком и моделью в памяти нужны CPU-сборка, smoke-аудио и веса `q4_0` и `m3_enc1` из первого запуска.
