#!/usr/bin/env bash
set -euo pipefail

FMT=${1:-q4_1}
case $FMT in
  q4_1) SUF= ;;
  q5_1) SUF=_q5 ;;
  *) echo "unknown format: $FMT" >&2; exit 1 ;;
esac
SIZES="20 50 200"
DATA=../../asr_data
OUT=../../results/ablation_$FMT
A=A$SUF.bin
B=B$SUF.bin

for f in ggml-model.bin "$A" "$B" stats.json stats_dev.npz $DATA/calib.txt $DATA/dev_en.tsv $DATA/dev_ru.tsv; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done
[ "$(wc -l < $DATA/calib.txt)" -eq 200 ] || { echo "calib.txt must have 200 lines" >&2; exit 1; }
MODEL=$(python -c 'import json; print(json.load(open("stats.json"))["model"])')

rm -rf "$OUT"
mkdir -p "$OUT"

MODELS=()
PAIRS=()
for n in $SIZES; do
  head -n "$n" $DATA/calib.txt > "calib_$n.txt"
  python ../calib/collect_stats.py --model "$MODEL" --audio-list "calib_$n.txt" --out "stats_cal$n"
  python ../calib/npz_to_imatrix.py "stats_cal$n.npz" "stats_cal$n.bin"
  cp "stats_cal$n.json" "$OUT/"

  C=C${SUF}_cal$n
  ./build/bin/whisper-quantize ggml-model.bin "$C.bin" "$FMT" --imatrix "stats_cal$n.bin" > "$C.log" 2>&1
  [ "$(grep -c '\[stats\]' "$C.log")" -eq 192 ] || { echo "$C: expected 192 [stats] tensors, see $C.log" >&2; exit 1; }

  python ../calib/layer_error.py --f16 ggml-model.bin --models "$A" "$B" "$C.bin" \
    --stats heldout=stats_dev.npz --out "$OUT/layer_cal$n"

  MODELS+=("$C.bin")
  PAIRS+=("B$SUF:$C" "A$SUF:$C")
done
PAIRS+=("C${SUF}_cal20:C${SUF}_cal200" "C${SUF}_cal50:C${SUF}_cal200")

for mode in gpu gpu_fallback; do
  EXTRA=()
  if [ "$mode" = gpu_fallback ]; then EXTRA=(--fallback); fi
  python ../calib/evaluate.py --cli ./build/bin/whisper-cli --hf ../../whisper-small \
    --models ggml-model.bin "$A" "$B" "${MODELS[@]}" --compare "${PAIRS[@]}" \
    --manifests $DATA/dev_en.tsv $DATA/dev_ru.tsv --out "$OUT/$mode" "${EXTRA[@]}"
done
echo "ALL DONE"
