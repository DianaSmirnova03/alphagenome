#!/usr/bin/env bash
# E2E head fine-tune: promoter (GPU 3) + immune (GPU 0), без npz X / seq_cache.
set -euo pipefail
BF="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${BF}/shared:${BF}/finetune_promoter:${BF}/finetune_immune:${BF}/tools"
cd "$BF"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ag
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88

# Данные и Stage-2 чекпоинты (пути на calc; при переносе — поменяйте здесь)
DATA_ROOT="/mnt/calc/homes/d.smirnova/DIPLOM"
FASTA="${DATA_ROOT}/AG/hg38.fa"
GTF="${DATA_ROOT}/alphagenome_snp_finetune/gencode.v39.feather"
PROM_DATA="${DATA_ROOT}/AG/data"
CHR_TRAIN="${DATA_ROOT}/chr_data/chr_train.h5"
CHR_TEST="${DATA_ROOT}/chr_data/chr_test.h5"
VOCAB="${DATA_ROOT}/alphagenome_snp_finetune/features_immune/cell_type_vocab.json"
HEAD_P="${DATA_ROOT}/alphagenome_snp_finetune/runs/effect_head_v1/best.pkl"
HEAD_I="${DATA_ROOT}/alphagenome_snp_finetune/runs/effect_head_immune_v2/best.pkl"

RUNS="${BF}/runs"
mkdir -p "$RUNS"

# Параметры запуска (L40S ~46 GB VRAM) — см. README в finetune_promoter / finetune_immune
WINDOW=16384
PROMOTER_BATCH=8
PROMOTER_VAL_BATCH=16
PROMOTER_SCALER_BATCH=8
IMMUNE_BATCH=2
IMMUNE_SCALER_BATCH=2
LR=1e-3

run_promoter() {
  export CUDA_VISIBLE_DEVICES=3
  python finetune_promoter/finetune_e2e_head_promoter.py \
    --train-csv "${PROM_DATA}/train_variants1.csv" \
    --val-csv "${PROM_DATA}/val_variants1.csv" \
    --fasta "$FASTA" --gtf-feather "$GTF" \
    --out-dir "${RUNS}/e2e_head_promoter_v1" \
    --head-checkpoint "$HEAD_P" \
    --batch-size "$PROMOTER_BATCH" --val-batch-size "$PROMOTER_VAL_BATCH" \
    --scaler-batch-size "$PROMOTER_SCALER_BATCH" --lr "$LR" \
    2>&1 | tee "${RUNS}/e2e_head_promoter_v1.log"
  python finetune_promoter/evaluate_e2e_promoter.py \
    --checkpoint "${RUNS}/e2e_head_promoter_v1/best.pkl" \
    --val-csv "${PROM_DATA}/val_variants1.csv" \
    --test-csv "${PROM_DATA}/tableS1A.tsv" \
    --fasta "$FASTA" --gtf-feather "$GTF" --test-csv-sep $'\t' \
    2>&1 | tee "${RUNS}/e2e_eval_promoter.log"
  unset JAX_PLATFORMS
  export CUDA_VISIBLE_DEVICES=3
  python tools/make_figures.py --task promoter --e2e \
    --promoter-checkpoint "${RUNS}/e2e_head_promoter_v1/best.pkl" \
    --promoter-tb "${RUNS}/e2e_head_promoter_v1/tensorboard" \
    --promoter-out "${BF}/finetune_promoter/analysis" \
    --metrics-json "${RUNS}/test_metrics_e2e.json" \
    --fasta "$FASTA" --gtf-feather "$GTF" \
    --promoter-val-csv "${PROM_DATA}/val_variants1.csv" \
    --promoter-test-csv "${PROM_DATA}/tableS1A.tsv"
}

run_immune() {
  export CUDA_VISIBLE_DEVICES=0
  python finetune_immune/finetune_e2e_head_immune.py \
    --h5 "$CHR_TRAIN" \
    --fasta "$FASTA" \
    --cell-type-vocab "$VOCAB" \
    --out-dir "${RUNS}/e2e_head_immune_v2" \
    --head-checkpoint "$HEAD_I" \
    --window "$WINDOW" --batch-size "$IMMUNE_BATCH" --scaler-batch-size "$IMMUNE_SCALER_BATCH" --lr "$LR" \
    2>&1 | tee "${RUNS}/e2e_head_immune_v2.log"
  python finetune_immune/evaluate_e2e_immune.py \
    --checkpoint "${RUNS}/e2e_head_immune_v2/best.pkl" \
    --train-h5 "$CHR_TRAIN" --test-h5 "$CHR_TEST" \
    --fasta "$FASTA" --cell-type-vocab "$VOCAB" \
    2>&1 | tee "${RUNS}/e2e_eval_immune.log"
  unset JAX_PLATFORMS
  export CUDA_VISIBLE_DEVICES=0
  python tools/make_figures.py --task immune --e2e \
    --immune-v2-checkpoint "${RUNS}/e2e_head_immune_v2/best.pkl" \
    --immune-tb "${RUNS}/e2e_head_immune_v2/tensorboard" \
    --immune-out "${BF}/finetune_immune/analysis_v2" \
    --metrics-json "${RUNS}/test_metrics_e2e.json" \
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
