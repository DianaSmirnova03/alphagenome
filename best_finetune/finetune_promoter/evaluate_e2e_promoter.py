"""E2E eval promoter: val/test через forward (CSV + FASTA + GTF), без npz X."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

from common import build_differentiable_apply_fn, get_rna_seq_valid_mask, load_alphagenome_model
from e2e_lfc import compute_rna_lfc_gene_mask, sanitize_lfc
from evaluate import LABEL_MAP, auc_pair
from extract_features import CONSEQUENCE_TO_LABEL
from heads import forward
from promoter_sequence_loader import filter_promoter_rows, load_promoter_csv, PromoterSequenceLoader


def load_head(checkpoint: str):
    ckpt = pickle.load(open(checkpoint, "rb"))
    return ckpt["params"], ckpt["config"], ckpt["scaler_mean"], ckpt["scaler_std"]


def build_predictor(checkpoint: str, fasta: str, gtf: str, window: int = 16384):
    from alphagenome.models.dna_model import Organism
    from alphagenome_research.model import dna_model as ag_dna_model

    params, config, mean, std = load_head(checkpoint)
    model = load_alphagenome_model(fasta_path=fasta, gtf_feather_path=gtf)
    backbone = model._params
    state = model._state
    _, apply_fn, _ = build_differentiable_apply_fn(model)
    valid_mask = jnp.asarray(get_rna_seq_valid_mask(model))
    org_idx = int(ag_dna_model.convert_to_organism_index(Organism.HOMO_SAPIENS))
    mean_j, std_j = jnp.asarray(mean), jnp.asarray(std)
    loader = PromoterSequenceLoader(fasta, gtf, window=window)

    @jax.jit
    def predict_z(ref, alt, gene_mask):
        lfc = compute_rna_lfc_gene_mask(
            apply_fn, backbone, state, ref, alt, gene_mask, valid_mask, org_idx
        )
        x = sanitize_lfc((lfc - mean_j) / std_j, clip=50.0)
        return forward(params, x, config, train=False)["z"]

    def run_df(df: pd.DataFrame, indices: np.ndarray, batch_size: int = 8):
        preds = []
        for s in range(0, len(indices), batch_size):
            bi = indices[s : s + batch_size]
            ref, alt, gm = loader.stack_batch_from_df(df, bi)
            preds.append(np.asarray(predict_z(jnp.asarray(ref), jnp.asarray(alt), jnp.asarray(gm))))
        return np.concatenate(preds) if preds else np.array([])

    return loader, run_df


def collect_promoter_e2e_predictions(
    checkpoint: str,
    val_csv: str,
    test_csv: str,
    fasta: str,
    gtf: str,
    *,
    csv_sep: str = ",",
    test_sep: str = "\t",
    batch_size: int = 8,
    window: int = 16384,
) -> dict:
    """Val/test pred для make_figures (E2E forward)."""
    loader, run_df = build_predictor(checkpoint, fasta, gtf, window=window)
    val_df = load_promoter_csv(val_csv, csv_sep=csv_sep)
    val_rows = filter_promoter_rows(val_df, loader)
    val_pred = run_df(val_df, val_rows, batch_size)
    val_z = val_df.iloc[val_rows]["z"].to_numpy(dtype=np.float32)

    test_df = load_promoter_csv(test_csv, csv_sep=test_sep)
    test_df = test_df.copy()
    test_df["_label"] = test_df["consequence"].map(CONSEQUENCE_TO_LABEL)
    test_rows = filter_promoter_rows(test_df, loader)
    test_pred = run_df(test_df, test_rows, batch_size)
    y_true = test_df.iloc[test_rows]["_label"].to_numpy(dtype=np.int64)
    valid = y_true >= 0
    return {
        "val_pred": val_pred,
        "val_z": val_z,
        "y_true": y_true[valid],
        "y_score": test_pred[valid],
    }


def evaluate(
    checkpoint: str,
    val_csv: str,
    test_csv: str,
    fasta: str,
    gtf: str,
    *,
    csv_sep: str = ",",
    test_sep: str = "\\t",
    batch_size: int = 8,
):
    loader, run_df = build_predictor(checkpoint, fasta, gtf)
    val_df = load_promoter_csv(val_csv, csv_sep=csv_sep)
    val_rows = filter_promoter_rows(val_df, loader)
    val_pred = run_df(val_df, val_rows, batch_size)
    val_z = val_df.iloc[val_rows]["z"].to_numpy(dtype=np.float32)
    mask = np.isfinite(val_z)
    r, _ = stats.pearsonr(val_pred[mask], val_z[mask])
    print("=== Val (E2E forward, CSV) ===")
    print(f"Pearson r = {r:.4f}  n={mask.sum()}")

    test_df = load_promoter_csv(test_csv, csv_sep=test_sep)
    if "consequence" not in test_df.columns:
        raise ValueError("test CSV needs consequence column")
    test_df = test_df.copy()
    test_df["_label"] = test_df["consequence"].map(CONSEQUENCE_TO_LABEL)
    test_rows = filter_promoter_rows(test_df, loader)
    test_pred = run_df(test_df, test_rows, batch_size)
    y_true = test_df.iloc[test_rows]["_label"].to_numpy(dtype=np.int64)
    valid = y_true >= 0
    y_true = y_true[valid]
    y_score = test_pred[valid]
    print("\n=== Test tableS1A (E2E forward) ===")
    print(f"over vs none  AUC = {auc_pair(y_true, y_score, LABEL_MAP['over'], LABEL_MAP['none']):.4f}")
    print(f"under vs none AUC = {auc_pair(y_true, y_score, LABEL_MAP['under'], LABEL_MAP['none']):.4f}")
    print(f"over vs under AUC = {auc_pair(y_true, y_score, LABEL_MAP['over'], LABEL_MAP['under']):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--gtf-feather", required=True)
    p.add_argument("--csv-sep", default=",")
    p.add_argument("--test-csv-sep", default="\\t")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    evaluate(
        args.checkpoint,
        args.val_csv,
        args.test_csv,
        args.fasta,
        args.gtf_feather,
        csv_sep=args.csv_sep,
        test_sep=args.test_csv_sep,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
