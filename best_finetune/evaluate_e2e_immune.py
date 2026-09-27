"""E2E eval immune: chr14 val + chr_test.h5 через forward, без npz X."""

from __future__ import annotations

import argparse
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from common import build_differentiable_apply_fn, get_rna_seq_valid_mask, load_alphagenome_model, normalize_chrom
from e2e_lfc import compute_rna_lfc_center_mask, sanitize_lfc
from extract_features_immune import _load_h5
from finetune_e2e_head_immune import _cell_type_indices
from finetune_lora_immune import _dedupe_batch_by_variant
from heads import forward
from immune_sequence_loader import ImmuneSequenceLoader
from train_immune import pearson_r, spearman_r
from train_immune_v2 import direction_auc


def collect_immune_e2e_predictions(
    checkpoint: str,
    train_h5: str,
    test_h5: str,
    fasta: str,
    cell_type_vocab_path: str,
    *,
    val_chrom: str = "14",
    window: int = 16384,
    center_width: int = 2001,
    batch_size: int = 4,
) -> dict:
    from alphagenome.models.dna_model import Organism
    from alphagenome_research.model import dna_model as ag_dna_model

    ckpt = pickle.load(open(checkpoint, "rb"))
    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]
    num_cell_types = ckpt["num_cell_types"]
    vocab = ckpt.get("cell_type_vocab") or json.loads(open(cell_type_vocab_path).read())

    model = load_alphagenome_model(fasta_path=fasta)
    backbone = model._params
    state = model._state
    _, apply_fn, _ = build_differentiable_apply_fn(model)
    valid_mask = jnp.asarray(get_rna_seq_valid_mask(model))
    org_idx = int(ag_dna_model.convert_to_organism_index(Organism.HOMO_SAPIENS))
    mean_j, std_j = jnp.asarray(mean), jnp.asarray(std)
    seq_loader = ImmuneSequenceLoader(fasta, window=window, center_width=center_width)

    train_df = _load_h5(train_h5)
    test_df = _load_h5(test_h5)
    val_rows = np.flatnonzero(train_df["chrom"].to_numpy(dtype=str) == normalize_chrom(val_chrom))
    test_rows = np.arange(len(test_df))

    def predict_rows(df, row_indices):
        chrom_l = df["chrom"].to_numpy(dtype=str)
        pos_l = df["pos"].to_numpy(dtype=np.int64)
        ref_l = df["ref"].to_numpy(dtype=str)
        alt_l = df["alt"].to_numpy(dtype=str)
        ct_all = _cell_type_indices(df["cell_type"].to_numpy(dtype=str), vocab)
        preds = []
        for s in range(0, len(row_indices), batch_size):
            ri = row_indices[s : s + batch_size]
            ukeys, u_index = _dedupe_batch_by_variant(ri, chrom_l, pos_l, ref_l, alt_l)
            ref_oh, alt_oh, masks = seq_loader.stack_batch(
                [k[0] for k in ukeys], [k[1] for k in ukeys], [k[2] for k in ukeys], [k[3] for k in ukeys]
            )
            lfc_u = compute_rna_lfc_center_mask(
                apply_fn, backbone, state,
                jnp.asarray(ref_oh), jnp.asarray(alt_oh), jnp.asarray(masks),
                valid_mask, org_idx,
            )
            x = sanitize_lfc((lfc_u[u_index] - mean_j) / std_j, clip=50.0)
            x = jnp.concatenate([x, jax.nn.one_hot(ct_all[ri].astype(jnp.int32), num_cell_types)], -1)
            preds.append(np.asarray(forward(params, x, config, train=False)["z"]))
        return np.concatenate(preds)

    return {
        "val_pred": predict_rows(train_df, val_rows),
        "val_z": train_df.iloc[val_rows]["comb_es"].to_numpy(dtype=np.float32),
        "val_n": int(len(val_rows)),
        "test_pred": predict_rows(test_df, test_rows),
        "test_z": test_df["comb_es"].to_numpy(dtype=np.float32),
        "test_fdr": test_df["fdr_comb_pval"].to_numpy(dtype=np.float32),
        "test_cell_type_idx": _cell_type_indices(test_df["cell_type"].to_numpy(dtype=str), vocab),
        "cell_type_vocab": vocab,
    }


def evaluate(
    checkpoint: str,
    train_h5: str,
    test_h5: str,
    fasta: str,
    *,
    cell_type_vocab_path: str,
    val_chrom: str = "14",
    window: int = 16384,
    center_width: int = 2001,
    batch_size: int = 4,
):
    data = collect_immune_e2e_predictions(
        checkpoint, train_h5, test_h5, fasta, cell_type_vocab_path,
        val_chrom=val_chrom, window=window, center_width=center_width, batch_size=batch_size,
    )
    val_pred, val_z = data["val_pred"], data["val_z"]
    test_pred, test_z = data["test_pred"], data["test_z"]
    test_fdr = data["test_fdr"]
    sig = test_fdr < 0.05
    print("=== Val chr14 (E2E forward) ===")
    print(f"Pearson r = {pearson_r(val_pred, val_z):.4f}  n={data['val_n']}")
    print("\n=== Test chr_test.h5 (E2E forward) ===")
    print(f"Pearson r  = {pearson_r(test_pred, test_z):.4f}")
    print(f"Spearman r = {spearman_r(test_pred, test_z):.4f}")
    print(f"Direction AUC (all) = {direction_auc(test_pred, test_z):.4f}")
    print(f"Direction AUC (fdr<0.05) = {direction_auc(test_pred[sig], test_z[sig]):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-h5", required=True)
    p.add_argument("--test-h5", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--cell-type-vocab", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    evaluate(
        args.checkpoint,
        args.train_h5,
        args.test_h5,
        args.fasta,
        cell_type_vocab_path=args.cell_type_vocab,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
