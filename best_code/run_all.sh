#!/usr/bin/env bash
# Полный прогон пайплайна Stage 1+2: извлечение признаков -> обучение -> оценка.
# Использует только GPU 0 (CUDA_VISIBLE_DEVICES=0).
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0

FASTA=/mnt/calc/homes/d.smirnova/DIPLOM/AG/hg38.fa
GTF_FEATHER=gencode.v39.feather
DATA_DIR=/mnt/calc/homes/d.smirnova/DIPLOM/AG/data
FEATURES_DIR=features
RUN_DIR=runs/effect_head_v1

mkdir -p "$FEATURES_DIR" "$RUN_DIR"

echo "=== [1/5] Извлечение признаков: train ==="
python extract_features.py \
    --csv "$DATA_DIR/train_variants1.csv" --split train \
    --fasta "$FASTA" --gtf-feather "$GTF_FEATHER" \
    --out-dir "$FEATURES_DIR"

echo "=== [2/5] Извлечение признаков: val ==="
python extract_features.py \
    --csv "$DATA_DIR/val_variants1.csv" --split val \
    --fasta "$FASTA" --gtf-feather "$GTF_FEATHER" \
    --out-dir "$FEATURES_DIR"

echo "=== [3/5] Извлечение признаков: test (tableS1A) ==="
python extract_features.py \
    --csv "$DATA_DIR/tableS1A.tsv" --split test --csv-sep '\t' \
    --fasta "$FASTA" --gtf-feather "$GTF_FEATHER" \
    --out-dir "$FEATURES_DIR"

echo "=== [4/5] Обучение головы ==="
python train.py --features-dir "$FEATURES_DIR" --out-dir "$RUN_DIR"

echo "=== [5/5] Оценка ==="
python evaluate.py --features-dir "$FEATURES_DIR" --checkpoint "$RUN_DIR/best.pkl"

echo "=== ГОТОВО ==="
