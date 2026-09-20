#!/usr/bin/env bash
# Полный пайплайн иммунных ASE-данных (Stage 1+2, рекомендуемая версия v2).
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FASTA="${FASTA:-/path/to/hg38.fa}"
CHR_TRAIN="${CHR_TRAIN:-../chr_data/chr_train.h5}"
CHR_TEST="${CHR_TEST:-../chr_data/chr_test.h5}"
OUT_FEATURES="${OUT_FEATURES:-features_immune}"
OUT_RUN="${OUT_RUN:-runs/effect_head_immune_v2}"
OUT_PLOTS="${OUT_PLOTS:-analysis_immune_v2}"

echo "=== 1. Экстракция LFC-признаков (CenterMask, RNA_SEQ) ==="
python extract_features_immune.py --h5 "$CHR_TRAIN" --split train_all \
    --fasta "$FASTA" --out-dir "$OUT_FEATURES" \
    --vocab-path "$OUT_FEATURES/cell_type_vocab.json"
python extract_features_immune.py --h5 "$CHR_TEST" --split test \
    --fasta "$FASTA" --out-dir "$OUT_FEATURES" \
    --vocab-path "$OUT_FEATURES/cell_type_vocab.json"

echo "=== 2. Обучение EffectHead v2 (FDR-веса + direction aux) ==="
python train_immune_v2.py --features-dir "$OUT_FEATURES" --out-dir "$OUT_RUN"

echo "=== 3. Оценка и графики ==="
python evaluate_immune_v2.py --features-dir "$OUT_FEATURES" \
    --checkpoint "$OUT_RUN/best.pkl" --plots-dir "$OUT_PLOTS"

echo "Готово. Чекпоинт: $OUT_RUN/best.pkl, графики: $OUT_PLOTS/"
