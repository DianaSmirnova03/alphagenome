"""Promoter E2E: голова поверх frozen AlphaGenome, данные только CSV + FASTA + GTF.

Без seq_cache и без npz-признаков X: LFC и scaler — онлайн через GPU.
Таргет z — из CSV. Опционально warm-start **только весов** головы из Stage 2.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter

from common import build_differentiable_apply_fn, get_rna_seq_valid_mask, load_alphagenome_model, set_seed
from e2e_lfc import compute_rna_lfc_gene_mask, sanitize_lfc
from e2e_scaler import fit_lfc_scaler_from_rows
from heads import HeadConfig, compute_loss, forward, init_params
from promoter_sequence_loader import filter_promoter_rows, load_promoter_csv, PromoterSequenceLoader
from train import make_batches, pearson_r


def train(
    train_csv: str,
    val_csv: str,
    fasta_path: str,
    gtf_feather_path: str,
    out_dir: str,
    *,
    csv_sep: str = ",",
    head_checkpoint: str | None = None,
    init_scaler_from_checkpoint: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    val_batch_size: int = 16,
    scaler_batch_size: int = 8,
    max_epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    tensorboard: bool = True,
    window: int = 16384,
):
    from alphagenome.models.dna_model import Organism
    from alphagenome_research.model import dna_model as ag_dna_model

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(out_dir / "tensorboard")) if tensorboard else None
    if tb is not None:
        print(f"TensorBoard-логи: {out_dir / 'tensorboard'}")

    train_df = load_promoter_csv(train_csv, csv_sep=csv_sep)
    val_df = load_promoter_csv(val_csv, csv_sep=csv_sep)
    loader = PromoterSequenceLoader(fasta_path, gtf_feather_path, window=window)
    print("Фильтр train (gene mask)...")
    train_rows = filter_promoter_rows(train_df, loader)
    print("Фильтр val...")
    val_rows = filter_promoter_rows(val_df, loader)
    print(f"Promoter E2E: train={len(train_rows)}/{len(train_df)} val={len(val_rows)}/{len(val_df)}")

    model = load_alphagenome_model(fasta_path=fasta_path, gtf_feather_path=gtf_feather_path)
    backbone = model._params
    state = model._state
    _, apply_fn, _ = build_differentiable_apply_fn(model)
    valid_mask = jnp.asarray(get_rna_seq_valid_mask(model))
    org_idx = int(ag_dna_model.convert_to_organism_index(Organism.HOMO_SAPIENS))

    head_config = HeadConfig(in_dim=int(valid_mask.sum()), predict_aux_p_over=False)
    key = jax.random.PRNGKey(seed)
    key, hk = jax.random.split(key)
    head_params = init_params(hk, head_config)
    mean = std = None
    if head_checkpoint and Path(head_checkpoint).exists():
        ckpt = pickle.load(open(head_checkpoint, "rb"))
        if ckpt["config"].in_dim == head_config.in_dim:
            head_params = ckpt["params"]
            print(f"Warm-start головы: {head_checkpoint}")
            if init_scaler_from_checkpoint:
                mean = np.asarray(ckpt["scaler_mean"])
                std = np.asarray(ckpt["scaler_std"])
                print("Scaler из чекпоинта (--init-scaler-from-checkpoint)")

    @jax.jit
    def raw_lfc(ref, alt, gene_mask):
        return compute_rna_lfc_gene_mask(
            apply_fn, backbone, state, ref, alt, gene_mask, valid_mask, org_idx
        )

    def stack_train(bi):
        ref, alt, gm = loader.stack_batch_from_df(train_df, bi)
        return ref, alt, gm

    if mean is None or std is None:
        print("Scaler по онлайн LFC на полном train...")
        mean, std = fit_lfc_scaler_from_rows(
            train_rows,
            batch_size=scaler_batch_size,
            lfc_rows_fn=lambda bi: np.asarray(
                raw_lfc(
                    jnp.asarray(stack_train(bi)[0]),
                    jnp.asarray(stack_train(bi)[1]),
                    jnp.asarray(stack_train(bi)[2]),
                )
            ),
            seed=seed,
        )
    np.savez(out_dir / "lfc_scaler.npz", mean=mean, std=std)

    params = head_params
    mean_j, std_j = jnp.asarray(mean), jnp.asarray(std)
    steps_per_epoch = max(1, -(-len(train_rows) // batch_size))
    total_steps = max(2, max_epochs * steps_per_epoch)
    warmup_steps = min(50, max(1, total_steps // 10))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.01,
    )
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=weight_decay))
    opt_state = opt.init(params)

    def scaled_lfc(ref, alt, gene_mask):
        return sanitize_lfc((raw_lfc(ref, alt, gene_mask) - mean_j) / std_j, clip=50.0)

    def loss_fn(p, batch, rng):
        x = scaled_lfc(batch["ref"], batch["alt"], batch["gene_mask"])
        loss, metrics = compute_loss(
            p,
            {"x": x, "z": batch["z"], "p_over": jnp.full_like(batch["z"], jnp.nan)},
            head_config,
            rng=rng,
            train=True,
            aux_weight=0.0,
        )
        return loss, metrics

    @jax.jit
    def train_step(p, opt_state, batch, rng):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, batch, rng)
        updates, opt_state = opt.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, metrics

    @jax.jit
    def predict_z(p, ref, alt, gene_mask):
        return forward(p, scaled_lfc(ref, alt, gene_mask), head_config, train=False)["z"]

    rng_np = np.random.default_rng(seed)
    best_r, best_p, stale = -np.inf, None, 0

    for epoch in range(max_epochs):
        losses = []
        for b in make_batches(len(train_rows), batch_size, rng_np):
            bi = train_rows[b]
            ref, alt, gm = stack_train(bi)
            z = train_df.iloc[bi]["z"].to_numpy(dtype=np.float32)
            batch = {
                "ref": jnp.asarray(ref),
                "alt": jnp.asarray(alt),
                "gene_mask": jnp.asarray(gm),
                "z": jnp.asarray(z),
            }
            key, sk = jax.random.split(key)
            params, opt_state, m = train_step(params, opt_state, batch, sk)
            losses.append(float(m["loss"]))

        preds, vz = [], []
        for s in range(0, len(val_rows), val_batch_size):
            bi = val_rows[s : s + val_batch_size]
            ref, alt, gm = loader.stack_batch_from_df(val_df, bi)
            preds.append(np.asarray(predict_z(params, jnp.asarray(ref), jnp.asarray(alt), jnp.asarray(gm))))
            vz.append(val_df.iloc[bi]["z"].to_numpy(dtype=np.float32))
        val_pred = np.concatenate(preds)
        val_z = np.concatenate(vz)
        val_r = pearson_r(val_pred, val_z)
        val_mse = float(np.nanmean((val_pred - val_z) ** 2))
        print(
            f"epoch {epoch:03d} | train_loss={np.mean(losses):.4f} |"
            f" val_pearson_r={val_r:.4f} | val_mse={val_mse:.4f}"
        )
        if tb is not None:
            tb.add_scalar("train/loss", float(np.mean(losses)), epoch)
            tb.add_scalar("val/pearson_r", float(val_r), epoch)
            tb.add_scalar("val/mse", float(val_mse), epoch)
            tb.add_scalar("train/lr", float(schedule(epoch * steps_per_epoch)), epoch)

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
                "best_val_pearson_r": best_r,
                "mode": "e2e_head_promoter",
                "train_csv": train_csv,
                "val_csv": val_csv,
            },
            f,
        )
    print(f"Сохранено {out_dir / 'best.pkl'} (best val_pearson_r={best_r:.4f})")
    if tb is not None:
        tb.add_hparams(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "mode": "e2e_head_promoter_online",
            },
            {"best_val_pearson_r": best_r},
        )
        tb.flush()
        tb.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--gtf-feather", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--csv-sep", default=",")
    p.add_argument("--head-checkpoint", default=None, help="Stage-2 веса головы (опционально)")
    p.add_argument("--init-scaler-from-checkpoint", action="store_true")
    p.add_argument("--window", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument("--scaler-batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-tensorboard", action="store_true")
    args = p.parse_args()
    train(
        args.train_csv,
        args.val_csv,
        args.fasta,
        args.gtf_feather,
        args.out_dir,
        csv_sep=args.csv_sep,
        head_checkpoint=args.head_checkpoint,
        init_scaler_from_checkpoint=args.init_scaler_from_checkpoint,
        window=args.window,
        lr=args.lr,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        scaler_batch_size=args.scaler_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        weight_decay=args.weight_decay,
        tensorboard=not args.no_tensorboard,
    )


if __name__ == "__main__":
    main()
