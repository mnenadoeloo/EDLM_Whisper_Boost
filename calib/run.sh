#!/usr/bin/env bash
set -euxo pipefail

OUT=../../results
DATA=../../asr_data
HF=../../whisper-small
CPUS=${CPUS:-8,9,10,11}
Q=./build/bin/whisper-quantize

for f in ggml-model.bin $DATA/calib.txt $DATA/dev_en.tsv $DATA/dev_ru.tsv; do
  if [ ! -f "$f" ]; then echo "missing $f" >&2; exit 1; fi
done

python ../calib/collect_stats.py --model $HF --audio-list $DATA/calib.txt --out stats
python ../calib/npz_to_imatrix.py stats.npz stats.bin
awk -F'\t' '{print $1"\t"$5}' $DATA/dev_en.tsv $DATA/dev_ru.tsv > dev_stats.txt
python ../calib/collect_stats.py --model $HF --audio-list dev_stats.txt --out stats_dev

$Q ggml-model.bin A.bin q4_1 > A.log 2>&1
$Q ggml-model.bin B.bin q4_1 --imatrix-ones > B.log 2>&1
$Q ggml-model.bin C.bin q4_1 --imatrix stats.bin > C.log 2>&1
$Q ggml-model.bin A_q5.bin q5_1 > A_q5.log 2>&1
$Q ggml-model.bin B_q5.bin q5_1 --imatrix-ones > B_q5.log 2>&1
$Q ggml-model.bin C_q5.bin q5_1 --imatrix stats.bin > C_q5.log 2>&1
for m in B B_q5; do test "$(grep -c '\[ones\]' $m.log)" = 193; done
for m in C C_q5; do test "$(grep -c '\[stats\]' $m.log)" = 192; done
python ../calib/test_axis.py --quantize $Q --f16 ggml-model.bin --hf $HF

mkdir -p $OUT
cp stats.json $OUT/
sha256sum ggml-model.bin A.bin B.bin C.bin A_q5.bin B_q5.bin C_q5.bin stats.bin > $OUT/sha256.txt
python ../calib/layer_error.py --f16 ggml-model.bin --models A.bin B.bin C.bin \
  --stats calib=stats.npz heldout=stats_dev.npz --out $OUT/layer_q4_1
python ../calib/layer_error.py --f16 ggml-model.bin --models A_q5.bin B_q5.bin C_q5.bin \
  --stats calib=stats.npz heldout=stats_dev.npz --out $OUT/layer_q5_1

MODELS="ggml-model.bin A.bin B.bin C.bin A_q5.bin B_q5.bin C_q5.bin"
PAIRS="A:B B:C A:C A_q5:B_q5 B_q5:C_q5 A_q5:C_q5 A:A_q5 B:B_q5 C:C_q5
       ggml-model:A ggml-model:B ggml-model:C ggml-model:A_q5 ggml-model:B_q5 ggml-model:C_q5"
EVAL="python ../calib/evaluate.py --cli ./build/bin/whisper-cli --hf $HF --manifests $DATA/dev_en.tsv $DATA/dev_ru.tsv"
$EVAL --models $MODELS --compare $PAIRS --out $OUT/gpu
$EVAL --models $MODELS --compare $PAIRS --out $OUT/gpu_fallback --fallback
$EVAL --models $MODELS --cpu --cpus $CPUS --limit 17 --repeats 3 --fallback --out $OUT/cpu

bash ../calib/ablation.sh

echo "ALL DONE"
