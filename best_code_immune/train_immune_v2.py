"""v2 обучения `EffectHead` на иммунных ASE-данных — по итогам анализа результатов v1.

Три изменения относительно `train_immune.py`, мотивированные разбором результатов
и сопоставлением с курсовой (раздел 4.3.1/5.1):

1. **Взвешивание регрессионного loss по `fdr_comb_pval`.** `comb_es` в строках с
   высоким FDR — это, по сути, оценка на шумных/малопокрытых данных (в курсовой
   такие строки на уровне сырых прочтений отбраковывались фильтром "покрытие >= 10
   ридов"; у нас такого фильтра нет, но fdr коррелирует с этим). Вес строки =
   exp(-3 * fdr) — уверенные ASE-вызовы (fdr~0) получают вес ~1, шумные (fdr~1) —
   вес ~0.05. Без переобучения экстракции признаков, это сразу поднимает эффективный
   SNR регрессии.
2. **Вторая (aux) голова классификации предсказывает DIRECTION (over/under по знаку
   `comb_es`), а не "значим ли ASE эффект" (`is_sig`).** `is_sig` зависит от
   статистической мощности (глубина покрытия, число клеток) — то, что модель
   в принципе не видит по последовательности, и его AUC на v1 был ~0.5 (случайность).
   `direction` — это ровно та задача, которую курсовая тестировала эмбеддингами +
   RF (AUC 0.763) — предсказуема по сигналу из последовательности.
3. Дополнительное логирование в TensorBoard: AUC(direction) на валидации по эпохам,
   Pearson r отдельно на подмножестве fdr<0.05 (наиболее надёжные метки).

Запуск:
    CUDA_VISIBLE_DEVICES=0 python train_immune_v2.py \
        --features-dir features_immune --out-dir runs/effect_head_immune_v2
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
from sklearn.metrics import roc_auc_score

from common import normalize_chrom, set_seed
from heads import HeadConfig, compute_loss, init_params, forward
from train_immune import (
    apply_scaler,
    build_input,
    fit_scaler,
    load_immune_split,
    make_batches,
    one_hot,
    pd_value_counts_top,
    pearson_r,
    spearman_r,
)


def direction_auc(pred: np.ndarray, comb_es: np.ndarray) -> float:
    """AUC для direction (1=under, 0=over по конвенции курсовой: comb_es>0 -> under)."""
    y = (comb_es > 0).astype(int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    auc = roc_auc_score(y, pred)
    return float(auc if auc >= 0.5 else 1.0 - auc)


def fdr_to_weight(fdr: np.ndarray, decay: float = 3.0, min_weight: float = 0.05) -> np.ndarray:
    fdr = np.nan_to_num(fdr, nan=1.0)
    w = np.exp(-decay * fdr)
    return np.maximum(w, min_weight).astype(np.float32)


def train(
    features_dir: str,
    out_dir: str,
    cell_type_vocab_path: str,
    val_chrom: str = "14",
    hidden_dims=(256, 64),
    dropout_rate: float = 0.2,
    aux_weight: float = 0.5,
    fdr_weight_decay: float = 3.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
    tensorboard: bool = True,
):
    devices = jax.devices()
    print(f"JAX видит устройства: {devices}")

    set_seed(seed)
    features_dir = Path(features_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_type_vocab = json.loads(Path(cell_type_vocab_path).read_text())
    num_cell_types = len(cell_type_vocab) + 1
    print(f"cell_type словарь: {len(cell_type_vocab)} типов (+1 unknown) = {num_cell_types} one-hot измерений")

    tb_writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = out_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard-логи: {tb_dir}")

    all_data = load_immune_split(features_dir, "train_all")
    chrom_str = np.asarray([c.decode() if isinstance(c, bytes) else str(c) for c in all_data["chrom"]])
    val_chrom_norm = normalize_chrom(val_chrom)
    val_mask = chrom_str == val_chrom_norm
    train_mask = ~val_mask
    print(
        f"train_all: {len(chrom_str)} строк -> train={train_mask.sum()} (все хромосомы кроме {val_chrom}),"
        f" val={val_mask.sum()} (хромосома {val_chrom})"
    )

    mean, std = fit_scaler(all_data["x_lfc"][train_mask])
    x_lfc_train = apply_scaler(all_data["x_lfc"][train_mask], mean, std)
    x_lfc_val = apply_scaler(all_data["x_lfc"][val_mask], mean, std)

    x_train = build_input(x_lfc_train, all_data["cell_type_idx"][train_mask], num_cell_types)
    x_val = build_input(x_lfc_val, all_data["cell_type_idx"][val_mask], num_cell_types)

    z_train = all_data["comb_es"][train_mask]
    z_val = all_data["comb_es"][val_mask]
    fdr_train = all_data["fdr_comb_pval"][train_mask]
    fdr_val = all_data["fdr_comb_pval"][val_mask]

    # NEW: aux target = direction (1=under, 0=over), не is_sig
    dir_train = (z_train > 0).astype(np.float32)
    dir_val = (z_val > 0).astype(np.float32)

    # NEW: per-row вес регрессии по уверенности ASE-вызова
    w_train = fdr_to_weight(fdr_train, decay=fdr_weight_decay)
    print(
        f"Веса по fdr: min={w_train.min():.3f} max={w_train.max():.3f} mean={w_train.mean():.3f}"
        f" (fdr<0.05 доля={float((fdr_train < 0.05).mean()):.4f})"
    )

    ct_val = all_data["cell_type_idx"][val_mask]

    config = HeadConfig(
        in_dim=x_train.shape[1],
        hidden_dims=tuple(hidden_dims),
        dropout_rate=dropout_rate,
        predict_aux_p_over=aux_weight > 0.0,
    )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, config)

    steps_per_epoch = max(1, -(-len(x_train) // batch_size))
    total_steps = max(2, max_epochs * steps_per_epoch)
    warmup_steps = min(200, max(1, total_steps // 10))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch, rng):
        def loss_fn(p):
            loss, metrics = compute_loss(p, batch, config, rng=rng, train=True, aux_weight=aux_weight)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {**metrics, "grad_norm": grad_norm}
        return params, opt_state, metrics

    @jax.jit
    def predict_all(params, x):
        out = forward(params, x, config, train=False)
        return out["z"], out.get("p_over_logit")

    rng_np = np.random.default_rng(seed)
    best_val_r = -np.inf
    best_state = None
    epochs_without_improvement = 0

    x_train_j = jnp.asarray(x_train)
    z_train_j = jnp.asarray(z_train)
    dir_train_j = jnp.asarray(dir_train)
    w_train_j = jnp.asarray(w_train)
    x_val_j = jnp.asarray(x_val)

    ct_names_val = np.asarray(cell_type_vocab + ["unknown"])[np.clip(ct_val, 0, len(cell_type_vocab))]
    top_cell_types = pd_value_counts_top(ct_names_val, k=10)
    fdr_sig_val_mask = fdr_val < 0.05

    for epoch in range(max_epochs):
        epoch_losses, epoch_z_losses, epoch_aux_losses, epoch_grad_norms = [], [], [], []
        for batch_idx in make_batches(len(x_train), batch_size, rng_np):
            key, step_key = jax.random.split(key)
            batch = {
                "x": x_train_j[batch_idx],
                "z": z_train_j[batch_idx],
                "p_over": dir_train_j[batch_idx],  # aux target = direction
                "sample_weight": w_train_j[batch_idx],
            }
            params, opt_state, metrics = train_step(params, opt_state, batch, step_key)
            epoch_losses.append(float(metrics["loss"]))
            epoch_z_losses.append(float(metrics["z_loss"]))
            if "aux_loss" in metrics:
                epoch_aux_losses.append(float(metrics["aux_loss"]))
            epoch_grad_norms.append(float(metrics["grad_norm"]))

        val_pred, val_dir_logit = predict_all(params, x_val_j)
        val_pred = np.asarray(val_pred)
        val_r = pearson_r(val_pred, z_val)
        val_rho = spearman_r(val_pred, z_val)
        val_mse = float(np.nanmean((val_pred - z_val) ** 2))
        val_r_sig_subset = pearson_r(val_pred[fdr_sig_val_mask], z_val[fdr_sig_val_mask])
        val_dir_auc = direction_auc(val_pred, z_val)

        print(
            f"epoch {epoch:03d} | train_loss={np.mean(epoch_losses):.4f} |"
            f" val_pearson_r={val_r:.4f} | val_pearson_r(fdr<0.05)={val_r_sig_subset:.4f} |"
            f" val_dir_auc={val_dir_auc:.4f} | val_mse={val_mse:.4f}"
        )

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", float(np.mean(epoch_losses)), epoch)
            tb_writer.add_scalar("train/z_loss", float(np.mean(epoch_z_losses)), epoch)
            if epoch_aux_losses:
                tb_writer.add_scalar("train/aux_loss_direction", float(np.mean(epoch_aux_losses)), epoch)
            tb_writer.add_scalar("train/grad_norm", float(np.mean(epoch_grad_norms)), epoch)
            tb_writer.add_scalar("train/lr", float(schedule(epoch * steps_per_epoch)), epoch)
            tb_writer.add_scalar("val/pearson_r", float(val_r), epoch)
            tb_writer.add_scalar("val/pearson_r_fdr_lt_0.05", float(val_r_sig_subset), epoch)
            tb_writer.add_scalar("val/spearman_r", float(val_rho), epoch)
            tb_writer.add_scalar("val/mse", float(val_mse), epoch)
            tb_writer.add_scalar("val/direction_auc", float(val_dir_auc), epoch)
            tb_writer.add_histogram("val/predictions", val_pred, epoch)
            tb_writer.add_histogram("val/targets", z_val, epoch)
            for ct in top_cell_types:
                sub = ct_names_val == ct
                if sub.sum() >= 5:
                    r_ct = pearson_r(val_pred[sub], z_val[sub])
                    tb_writer.add_scalar(f"val_by_cell_type/{ct}", float(r_ct), epoch)

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = jax.tree_util.tree_map(lambda a: np.array(a), params)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Ранняя остановка на эпохе {epoch} (best val_pearson_r={best_val_r:.4f})")
                break

    checkpoint = {
        "params": best_state,
        "config": config,
        "scaler_mean": mean,
        "scaler_std": std,
        "cell_type_vocab": cell_type_vocab,
        "num_cell_types": num_cell_types,
        "best_val_pearson_r": best_val_r,
        "aux_target": "direction",  # для evaluate_immune_v2.py
    }
    with open(out_dir / "best.pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Сохранено {out_dir / 'best.pkl'} (best val_pearson_r={best_val_r:.4f})")

    if tb_writer is not None:
        tb_writer.add_hparams(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout_rate": dropout_rate,
                "batch_size": batch_size,
                "hidden_dims": str(hidden_dims),
                "aux_weight": aux_weight,
                "fdr_weight_decay": fdr_weight_decay,
                "val_chrom": str(val_chrom),
            },
            {"best_val_pearson_r": best_val_r},
        )
        tb_writer.flush()
        tb_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cell-type-vocab", default=None)
    parser.add_argument("--val-chrom", default="14")
    parser.add_argument("--hidden-dims", default="256,64")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--fdr-weight-decay", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x)
    vocab_path = args.cell_type_vocab or str(Path(args.features_dir) / "cell_type_vocab.json")

    train(
        features_dir=args.features_dir,
        out_dir=args.out_dir,
        cell_type_vocab_path=vocab_path,
        val_chrom=args.val_chrom,
        hidden_dims=hidden_dims,
        dropout_rate=args.dropout_rate,
        aux_weight=args.aux_weight,
        fdr_weight_decay=args.fdr_weight_decay,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        tensorboard=not args.no_tensorboard,
    )


if __name__ == "__main__":
    main()
