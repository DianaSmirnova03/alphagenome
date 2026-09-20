#!/usr/bin/env bash
# Полный пайплайн на иммунных ASE-данных (chr_train.h5 / chr_test.h5).
# Строго одно GPU-ядро (по умолчанию 0 — задаётся снаружи через GPU_ID).
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

FASTA=/mnt/calc/homes/d.smirnova/DIPLOM/AG/hg38.fa
CHR_TRAIN=/mnt/calc/homes/d.smirnova/DIPLOM/chr_data/chr_train.h5
CHR_TEST=/mnt/calc/homes/d.smirnova/DIPLOM/chr_data/chr_test.h5
OUT_FEATURES=features_immune
OUT_RUN=runs/effect_head_immune_v1
OUT_PLOTS=analysis_immune

echo "=== GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

mkdir -p "$OUT_FEATURES"

echo "=== 1/3: извлечение LFC-признаков (CenterMaskScorer RNA_SEQ) ==="
python extract_features_immune.py --h5 "$CHR_TRAIN" --split train_all \
    --fasta "$FASTA" --out-dir "$OUT_FEATURES" \
    --vocab-path "$OUT_FEATURES/cell_type_vocab.json"

python extract_features_immune.py --h5 "$CHR_TEST" --split test \
    --fasta "$FASTA" --out-dir "$OUT_FEATURES" \
    --vocab-path "$OUT_FEATURES/cell_type_vocab.json"

echo "=== 2/3: обучение EffectHead (val = хромосома 14) ==="
python train_immune.py --features-dir "$OUT_FEATURES" --out-dir "$OUT_RUN" --val-chrom 14

echo "=== 3/3: оценка на held-out тесте + графики ==="
python evaluate_immune.py --features-dir "$OUT_FEATURES" --checkpoint "$OUT_RUN/best.pkl" --plots-dir "$OUT_PLOTS"

echo "Готово. TensorBoard: tensorboard --logdir $OUT_RUN/tensorboard"
