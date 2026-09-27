#!/usr/bin/env bash
# E2E head fine-tune: только CSV/H5 + FASTA, без npz-признаков и seq_cache.
set -euo pipefail
ROOT="/mnt/calc/homes/d.smirnova/DIPLOM/alphagenome_snp_finetune"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ag
unset JAX_PLATFORMS
# true = сразу занять большую долю VRAM (как extract/score_variant), соседям на GPU сложнее вклиниться
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88

FASTA="/mnt/calc/homes/d.smirnova/DIPLOM/AG/hg38.fa"
GTF="${ROOT}/gencode.v39.feather"
DATA="/mnt/calc/homes/d.smirnova/DIPLOM/AG/data"
CHR_TRAIN="/mnt/calc/homes/d.smirnova/DIPLOM/chr_data/chr_train.h5"
CHR_TEST="/mnt/calc/homes/d.smirnova/DIPLOM/chr_data/chr_test.h5"
VOCAB="${ROOT}/features_immune/cell_type_vocab.json"
WINDOW=16384
# E2E на L40S (46GB): крупный batch + PREALLOCATE ≈ занять ~40GB на карту (если OOM — уменьшите на 2)
PROMOTER_BATCH=8
PROMOTER_VAL_BATCH=16
PROMOTER_SCALER_BATCH=8
IMMUNE_BATCH=2
IMMUNE_SCALER_BATCH=2
mkdir -p "${ROOT}/runs"

run_promoter() {
  export CUDA_VISIBLE_DEVICES=3
  python finetune_e2e_head_promoter.py \
    --train-csv "${DATA}/train_variants1.csv" \
    --val-csv "${DATA}/val_variants1.csv" \
    --fasta "$FASTA" --gtf-feather "$GTF" \
    --out-dir "${ROOT}/runs/e2e_head_promoter_v1" \
    --head-checkpoint "${ROOT}/runs/effect_head_v1/best.pkl" \
    --batch-size "$PROMOTER_BATCH" --val-batch-size "$PROMOTER_VAL_BATCH" \
    --scaler-batch-size "$PROMOTER_SCALER_BATCH" --lr 1e-3 \
    2>&1 | tee "${ROOT}/runs/e2e_head_promoter_v1.log"
  export JAX_PLATFORMS=cpu
  export CUDA_VISIBLE_DEVICES=3
  python evaluate_e2e_promoter.py \
    --checkpoint "${ROOT}/runs/e2e_head_promoter_v1/best.pkl" \
    --val-csv "${DATA}/val_variants1.csv" \
    --test-csv "${DATA}/tableS1A.tsv" \
    --fasta "$FASTA" --gtf-feather "$GTF" --test-csv-sep $'\t' \
    2>&1 | tee "${ROOT}/runs/e2e_eval_promoter.log"
  unset JAX_PLATFORMS
  export CUDA_VISIBLE_DEVICES=3
  python make_figures.py --task promoter --e2e \
    --promoter-checkpoint runs/e2e_head_promoter_v1/best.pkl \
    --promoter-tb runs/e2e_head_promoter_v1/tensorboard \
    --promoter-out analysis \
    --metrics-json results_eval/test_metrics_e2e.json \
    --fasta "$FASTA" --gtf-feather "$GTF" \
    --promoter-val-csv "${DATA}/val_variants1.csv" \
    --promoter-test-csv "${DATA}/tableS1A.tsv"
}

run_immune() {
  export CUDA_VISIBLE_DEVICES=0
  python finetune_e2e_head_immune.py \
    --h5 "$CHR_TRAIN" \
    --fasta "$FASTA" \
    --cell-type-vocab "$VOCAB" \
    --out-dir "${ROOT}/runs/e2e_head_immune_v2" \
    --head-checkpoint "${ROOT}/runs/effect_head_immune_v2/best.pkl" \
    --window "$WINDOW" --batch-size "$IMMUNE_BATCH" --scaler-batch-size "$IMMUNE_SCALER_BATCH" --lr 1e-3 \
    2>&1 | tee "${ROOT}/runs/e2e_head_immune_v2.log"
  export JAX_PLATFORMS=cpu
  export CUDA_VISIBLE_DEVICES=0
  python evaluate_e2e_immune.py \
    --checkpoint "${ROOT}/runs/e2e_head_immune_v2/best.pkl" \
    --train-h5 "$CHR_TRAIN" --test-h5 "$CHR_TEST" \
    --fasta "$FASTA" --cell-type-vocab "$VOCAB" \
    2>&1 | tee "${ROOT}/runs/e2e_eval_immune.log"
  unset JAX_PLATFORMS
  export CUDA_VISIBLE_DEVICES=0
  python make_figures.py --task immune --e2e \
    --immune-v2-checkpoint runs/e2e_head_immune_v2/best.pkl \
    --immune-tb runs/e2e_head_immune_v2/tensorboard \
    --immune-out analysis_v2 \
    --metrics-json results_eval/test_metrics_e2e.json \
    --fasta "$FASTA" --train-h5 "$CHR_TRAIN" --test-h5 "$CHR_TEST" \
    --cell-type-vocab "$VOCAB"
}

case "${1:-both}" in
  promoter) run_promoter ;;
  immune) run_immune ;;
  both)
    run_immune &
    run_promoter &
    wait
    ;;
  *) echo "Usage: $0 {promoter|immune|both}"; exit 1 ;;
esac
