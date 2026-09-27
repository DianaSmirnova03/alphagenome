"""Immune E2E: голова поверх frozen AlphaGenome; данные chr_train.h5 + hg38 (без npz X).

Loss как v2. Scaler и LFC — онлайн. Warm-start только весов головы (опционально).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter

from common import build_differentiable_apply_fn, get_rna_seq_valid_mask, load_alphagenome_model, normalize_chrom, set_seed
from e2e_lfc import compute_rna_lfc_center_mask, sanitize_lfc
from e2e_scaler import fit_lfc_scaler_from_rows
from extract_features_immune import _load_h5
from e2e_batch_utils import dedupe_batch_by_variant as _dedupe_batch_by_variant, zero_nonfinite_grads as _zero_nonfinite_grads
from heads import HeadConfig, compute_loss, forward, init_params
from immune_sequence_loader import ImmuneSequenceLoader
from train_immune import make_batches, pd_value_counts_top, pearson_r, spearman_r
from train_immune_v2 import direction_auc, fdr_to_weight


def _cell_type_indices(cell_types: np.ndarray, vocab: list[str]) -> np.ndarray:
    idx_map = {c: i for i, c in enumerate(vocab)}
    unknown = len(vocab)
    return np.asarray([idx_map.get(str(c), unknown) for c in cell_types], dtype=np.int64)


def train(
    h5_path: str,
    fasta_path: str,
    out_dir: str,
    *,
    cell_type_vocab_path: str | None = None,
    head_checkpoint: str | None = None,
    init_scaler_from_checkpoint: bool = False,
    val_chrom: str = "14",
    window: int = 16384,
    center_width: int = 2001,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 2,
    scaler_batch_size: int = 4,
    max_epochs: int = 200,
    patience: int = 20,
    aux_weight: float = 0.5,
    fdr_weight_decay: float = 3.0,
    max_train_rows: int | None = None,
    seed: int = 42,
    tensorboard: bool = True,
):
    from alphagenome.models.dna_model import Organism
    from alphagenome_research.model import dna_model as ag_dna_model

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(tb_dir)) if tensorboard else None
    if tb is not None:
        print(f"TensorBoard-логи: {tb_dir}")

    h5_df = _load_h5(h5_path)
    if cell_type_vocab_path and Path(cell_type_vocab_path).exists():
        vocab = json.loads(Path(cell_type_vocab_path).read_text())
    else:
        vocab = sorted(h5_df["cell_type"].unique().tolist())
    num_cell_types = len(vocab) + 1

    chrom = h5_df["chrom"].to_numpy(dtype=str)
    pos = h5_df["pos"].to_numpy(dtype=np.int64)
    ref = h5_df["ref"].to_numpy(dtype=str)
    alt = h5_df["alt"].to_numpy(dtype=str)
    z_all = h5_df["comb_es"].to_numpy(dtype=np.float32)
    fdr_all = h5_df["fdr_comb_pval"].to_numpy(dtype=np.float32)
    ct_all = _cell_type_indices(h5_df["cell_type"].to_numpy(dtype=str), vocab)

    val_mask = chrom == normalize_chrom(val_chrom)
    train_mask = ~val_mask
    if max_train_rows:
        tr = np.flatnonzero(train_mask)
        keep = np.random.default_rng(seed).choice(tr, size=min(max_train_rows, len(tr)), replace=False)
        m = np.zeros(len(chrom), bool)
        m[keep] = True
        train_mask &= m
    train_rows = np.flatnonzero(train_mask)
    val_rows = np.flatnonzero(val_mask)
    print(f"E2E immune: train={len(train_rows)} val={len(val_rows)} (h5={h5_path})")

    seq_loader = ImmuneSequenceLoader(fasta_path, window=window, center_width=center_width)

    model = load_alphagenome_model(fasta_path=fasta_path)
    backbone = model._params
    state = model._state
    _, apply_fn, _ = build_differentiable_apply_fn(model)
    valid_mask = jnp.asarray(get_rna_seq_valid_mask(model))
    org_idx = int(ag_dna_model.convert_to_organism_index(Organism.HOMO_SAPIENS))

    @jax.jit
    def raw_lfc_center(ref, alt, center_mask):
        return compute_rna_lfc_center_mask(
            apply_fn, backbone, state, ref, alt, center_mask, valid_mask, org_idx
        )

    def lfc_rows(row_indices: np.ndarray) -> np.ndarray:
        ukeys, u_index = _dedupe_batch_by_variant(row_indices, chrom, pos, ref, alt)
        ref_oh, alt_oh, masks = seq_loader.stack_batch(
            [k[0] for k in ukeys], [k[1] for k in ukeys], [k[2] for k in ukeys], [k[3] for k in ukeys]
        )
        lfc_u = np.asarray(
            raw_lfc_center(jnp.asarray(ref_oh), jnp.asarray(alt_oh), jnp.asarray(masks))
        )
        return lfc_u[u_index]

    mean = std = None
    in_dim = int(valid_mask.sum()) + num_cell_types
    head_config = HeadConfig(in_dim=in_dim, predict_aux_p_over=aux_weight > 0)
    key = jax.random.PRNGKey(seed)
    key, hk = jax.random.split(key)
    head_params = init_params(hk, head_config)
    if head_checkpoint and Path(head_checkpoint).exists():
        ckpt = pickle.load(open(head_checkpoint, "rb"))
        if ckpt["config"].in_dim == in_dim:
            head_params = ckpt["params"]
            print(f"Warm-start головы: {head_checkpoint}")
            if init_scaler_from_checkpoint:
                mean = np.asarray(ckpt["scaler_mean"])
                std = np.asarray(ckpt["scaler_std"])

    if mean is None or std is None:
        print("Scaler по онлайн LFC (полный train, без npz)...")
        mean, std = fit_lfc_scaler_from_rows(
            train_rows, batch_size=scaler_batch_size, lfc_rows_fn=lfc_rows, seed=seed
        )
    np.savez(out_dir / "lfc_scaler.npz", mean=mean, std=std)

    fdr_train = fdr_all[train_mask]
    w_train = fdr_to_weight(fdr_train, decay=fdr_weight_decay)
    print(
        f"Веса по fdr: min={w_train.min():.3f} max={w_train.max():.3f} mean={w_train.mean():.3f}"
        f" (fdr<0.05 доля={float((fdr_train < 0.05).mean()):.4f})"
    )

    ct_val = ct_all[val_mask]
    ct_names_val = np.asarray(vocab + ["unknown"])[np.clip(ct_val, 0, len(vocab))]
    top_cell_types = pd_value_counts_top(ct_names_val, k=10)
    fdr_val = fdr_all[val_mask]
    z_val = z_all[val_mask]
    fdr_sig_val_mask = fdr_val < 0.05

    params = head_params
    steps_per_epoch = max(1, -(-len(train_rows) // batch_size))
    total_steps = max(2, max_epochs * steps_per_epoch)
    warmup_steps = min(200, max(1, total_steps // 10))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.01,
    )
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=weight_decay))
    opt_state = opt.init(params)
    mean_j, std_j = jnp.asarray(mean), jnp.asarray(std)

    w_all = fdr_to_weight(fdr_all, decay=fdr_weight_decay)
    dir_all = (z_all > 0).astype(np.float32)

    def head_input(lfc_rows, ct_idx):
        x = sanitize_lfc((lfc_rows - mean_j) / std_j, clip=50.0)
        return jnp.concatenate([x, jax.nn.one_hot(ct_idx.astype(jnp.int32), num_cell_types)], -1)

    def make_batch(row_indices):
        ukeys, u_index = _dedupe_batch_by_variant(row_indices, chrom, pos, ref, alt)
        ref_oh, alt_oh, masks = seq_loader.stack_batch(
            [k[0] for k in ukeys], [k[1] for k in ukeys], [k[2] for k in ukeys], [k[3] for k in ukeys]
        )
        return {
            "ref": jnp.asarray(ref_oh),
            "alt": jnp.asarray(alt_oh),
            "center_mask": jnp.asarray(masks),
            "u_index": jnp.asarray(u_index),
            "cell_type_idx": jnp.asarray(ct_all[row_indices]),
            "z": jnp.asarray(z_all[row_indices]),
            "direction": jnp.asarray(dir_all[row_indices]),
            "sample_weight": jnp.asarray(w_all[row_indices]),
        }

    def loss_fn(p, batch, rng):
        lfc_u = compute_rna_lfc_center_mask(
            apply_fn, backbone, state, batch["ref"], batch["alt"], batch["center_mask"], valid_mask, org_idx
        )
        x = head_input(lfc_u[batch["u_index"]], batch["cell_type_idx"])
        loss, metrics = compute_loss(
            p,
            {"x": x, "z": batch["z"], "p_over": batch["direction"], "sample_weight": batch["sample_weight"]},
            head_config,
            rng=rng,
            train=True,
            aux_weight=aux_weight,
        )
        return loss, metrics

    @jax.jit
    def train_step(p, opt_state, batch, rng):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, batch, rng)
        grads = _zero_nonfinite_grads(grads)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state, p)
        metrics = {**metrics, "grad_norm": grad_norm}
        return optax.apply_updates(p, updates), opt_state, metrics

    @jax.jit
    def predict_batch(p, batch):
        lfc_u = compute_rna_lfc_center_mask(
            apply_fn, backbone, state, batch["ref"], batch["alt"], batch["center_mask"], valid_mask, org_idx
        )
        x = head_input(lfc_u[batch["u_index"]], batch["cell_type_idx"])
        return forward(p, x, head_config, train=False)["z"]

    rng_np = np.random.default_rng(seed)
    best_r, best_p, stale = -np.inf, None, 0
    key = jax.random.PRNGKey(seed + 1)

    for epoch in range(max_epochs):
        epoch_losses, epoch_z_losses, epoch_aux_losses, epoch_grad_norms = [], [], [], []
        for bidx in make_batches(len(train_rows), batch_size, rng_np):
            batch = make_batch(train_rows[bidx])
            key, sk = jax.random.split(key)
            params, opt_state, m = train_step(params, opt_state, batch, sk)
            epoch_losses.append(float(m["loss"]))
            epoch_z_losses.append(float(m["z_loss"]))
            if "aux_loss" in m:
                epoch_aux_losses.append(float(m["aux_loss"]))
            epoch_grad_norms.append(float(m["grad_norm"]))

        vp, vz = [], []
        for s in range(0, len(val_rows), batch_size):
            ri = val_rows[s : s + batch_size]
            batch = make_batch(ri)
            vp.append(np.asarray(predict_batch(params, batch)))
            vz.append(z_all[ri])
        val_pred, val_z_ep = np.concatenate(vp), np.concatenate(vz)
        val_r = pearson_r(val_pred, val_z_ep)
        val_rho = spearman_r(val_pred, val_z_ep)
        val_mse = float(np.nanmean((val_pred - val_z_ep) ** 2))
        val_r_sig_subset = pearson_r(val_pred[fdr_sig_val_mask], z_val[fdr_sig_val_mask])
        val_auc = direction_auc(val_pred, val_z_ep)

        print(
            f"epoch {epoch:03d} | train_loss={np.mean(epoch_losses):.4f} |"
            f" val_pearson_r={val_r:.4f} | val_pearson_r(fdr<0.05)={val_r_sig_subset:.4f} |"
            f" val_dir_auc={val_auc:.4f} | val_mse={val_mse:.4f}"
        )

        if tb is not None:
            tb.add_scalar("train/loss", float(np.mean(epoch_losses)), epoch)
            tb.add_scalar("train/z_loss", float(np.mean(epoch_z_losses)), epoch)
            if epoch_aux_losses:
                tb.add_scalar("train/aux_loss_direction", float(np.mean(epoch_aux_losses)), epoch)
            tb.add_scalar("train/grad_norm", float(np.mean(epoch_grad_norms)), epoch)
            tb.add_scalar("train/lr", float(schedule(epoch * steps_per_epoch)), epoch)
            tb.add_scalar("val/pearson_r", float(val_r), epoch)
            tb.add_scalar("val/pearson_r_fdr_lt_0.05", float(val_r_sig_subset), epoch)
            tb.add_scalar("val/spearman_r", float(val_rho), epoch)
            tb.add_scalar("val/mse", float(val_mse), epoch)
            tb.add_scalar("val/direction_auc", float(val_auc), epoch)
            tb.add_histogram("val/predictions", val_pred, epoch)
            tb.add_histogram("val/targets", val_z_ep, epoch)
            for ct in top_cell_types:
                sub = ct_names_val == ct
                if sub.sum() >= 5:
                    r_ct = pearson_r(val_pred[sub], z_val[sub])
                    tb.add_scalar(f"val_by_cell_type/{ct}", float(r_ct), epoch)

        if val_r > best_r:
            best_r, best_p, stale = val_r, jax.tree_util.tree_map(np.array, params), 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Ранняя остановка на эпохе {epoch} (best val_pearson_r={best_r:.4f})")
                break

    with open(out_dir / "best.pkl", "wb") as f:
        pickle.dump(
            {
                "params": best_p,
                "config": head_config,
                "scaler_mean": mean,
                "scaler_std": std,
                "cell_type_vocab": vocab,
                "num_cell_types": num_cell_types,
                "best_val_pearson_r": best_r,
                "aux_target": "direction",
                "mode": "e2e_head_immune_v2",
            },
            f,
        )
    print(f"Сохранено {out_dir / 'best.pkl'} (best val_pearson_r={best_r:.4f})")
    if tb is not None:
        tb.add_hparams(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout_rate": head_config.dropout_rate,
                "batch_size": batch_size,
                "hidden_dims": str(head_config.hidden_dims),
                "aux_weight": aux_weight,
                "fdr_weight_decay": fdr_weight_decay,
                "val_chrom": str(val_chrom),
                "mode": "e2e_head_immune",
            },
            {"best_val_pearson_r": best_r},
        )
        tb.flush()
        tb.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, help="chr_train.h5")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--cell-type-vocab",
        default="/mnt/calc/homes/d.smirnova/DIPLOM/alphagenome_snp_finetune/features_immune/cell_type_vocab.json",
        help="JSON словарь cell_type (метки из h5)",
    )
    p.add_argument("--head-checkpoint", default=None)
    p.add_argument("--init-scaler-from-checkpoint", action="store_true")
    p.add_argument("--window", type=int, default=16384)
    p.add_argument("--center-width", type=int, default=2001)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--scaler-batch-size", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--fdr-weight-decay", type=float, default=3.0)
    p.add_argument("--no-tensorboard", action="store_true")
    args = p.parse_args()
    train(
        args.h5,
        args.fasta,
        args.out_dir,
        cell_type_vocab_path=args.cell_type_vocab,
        head_checkpoint=args.head_checkpoint,
        init_scaler_from_checkpoint=args.init_scaler_from_checkpoint,
        window=args.window,
        center_width=args.center_width,
        lr=args.lr,
        batch_size=args.batch_size,
        scaler_batch_size=args.scaler_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        aux_weight=args.aux_weight,
        max_train_rows=args.max_train_rows,
        weight_decay=args.weight_decay,
        fdr_weight_decay=args.fdr_weight_decay,
        tensorboard=not args.no_tensorboard,
    )


if __name__ == "__main__":
    main()
