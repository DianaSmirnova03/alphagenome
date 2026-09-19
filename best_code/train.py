"""Обучение `EffectHead` на закэшированных признаках (`extract_features.py`).

Лосс: Huber на `z` (робастный к тяжёлым хвостам — max|z| ~ 8.3, std ~ 0.45)
+ опциональная вспомогательная BCE-голова на `p_over` (мягкая метка,
`--aux-weight > 0`, по умолчанию выключена, чтобы численно точно совпадать
с постановкой задачи "предсказать z").

Ранняя остановка — по корреляции Пирсона между предсказанным и истинным `z`
на валидации (та же метрика, что показывала переобучение во всех старых
запусках — здесь она должна расти, а не падать).

Запуск:
    python train.py --features-dir features --out-dir runs/effect_head_v1
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import stats

from common import set_seed
from heads import HeadConfig, compute_loss, init_params


def load_split(features_dir: Path, split: str):
    data = np.load(features_dir / f"{split}_features.npz")
    return {
        "x": data["X"].astype(np.float32),
        "z": data["z"].astype(np.float32),
        "p_over": data["p_over"].astype(np.float32),
        "p_under": data["p_under"].astype(np.float32),
        "consequence": data["consequence"].astype(np.int64),
    }


def fit_scaler(x: np.ndarray):
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    return (x - mean) / std


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(target)
    if mask.sum() < 2:
        return float("nan")
    r, _ = stats.pearsonr(pred[mask], target[mask])
    return float(r)


def make_batches(n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def train(
    features_dir: str,
    out_dir: str,
    hidden_dims=(256, 64),
    dropout_rate: float = 0.2,
    aux_weight: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    tensorboard: bool = True,
):
    set_seed(seed)
    features_dir = Path(features_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = None
    if tensorboard:
        # `torch.utils.tensorboard.SummaryWriter` пишет обычные TF-совместимые
        # event-файлы, не требуя полноценного TensorFlow-обучения — удобно в
        # JAX-окружении. Логи кладём в <out_dir>/tensorboard, смотреть их
        # можно `tensorboard --logdir <out_dir>/tensorboard` (см. README).
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = out_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard-логи: {tb_dir}")

    train_data = load_split(features_dir, "train")
    val_data = load_split(features_dir, "val")

    mean, std = fit_scaler(train_data["x"])
    x_train = apply_scaler(train_data["x"], mean, std)
    x_val = apply_scaler(val_data["x"], mean, std)

    config = HeadConfig(
        in_dim=x_train.shape[1],
        hidden_dims=tuple(hidden_dims),
        dropout_rate=dropout_rate,
        predict_aux_p_over=aux_weight > 0.0,
    )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, config)

    steps_per_epoch = max(1, -(-len(x_train) // batch_size))  # ceil division
    total_steps = max(2, max_epochs * steps_per_epoch)
    warmup_steps = min(50, max(1, total_steps // 10))
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
            loss, metrics = compute_loss(
                p, batch, config, rng=rng, train=True, aux_weight=aux_weight
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def predict_z(params, x):
        from heads import forward

        return forward(params, x, config, train=False)["z"]

    rng_np = np.random.default_rng(seed)
    best_val_r = -np.inf
    best_state = None
    epochs_without_improvement = 0

    x_train_j = jnp.asarray(x_train)
    z_train_j = jnp.asarray(train_data["z"])
    p_over_train_j = jnp.asarray(train_data["p_over"])
    x_val_j = jnp.asarray(x_val)

    for epoch in range(max_epochs):
        epoch_losses = []
        for batch_idx in make_batches(len(x_train), batch_size, rng_np):
            key, step_key = jax.random.split(key)
            batch = {
                "x": x_train_j[batch_idx],
                "z": z_train_j[batch_idx],
                "p_over": p_over_train_j[batch_idx],
            }
            params, opt_state, metrics = train_step(params, opt_state, batch, step_key)
            epoch_losses.append(float(metrics["loss"]))

        val_pred = np.asarray(predict_z(params, x_val_j))
        val_r = pearson_r(val_pred, val_data["z"])
        val_mse = float(np.nanmean((val_pred - val_data["z"]) ** 2))

        print(
            f"epoch {epoch:03d} | train_loss={np.mean(epoch_losses):.4f} |"
            f" val_pearson_r={val_r:.4f} | val_mse={val_mse:.4f}"
        )

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", float(np.mean(epoch_losses)), epoch)
            tb_writer.add_scalar("val/pearson_r", float(val_r), epoch)
            tb_writer.add_scalar("val/mse", float(val_mse), epoch)
            tb_writer.add_scalar("train/lr", float(schedule(epoch * max(1, len(x_train) // batch_size))), epoch)

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
        "best_val_pearson_r": best_val_r,
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
            },
            {"best_val_pearson_r": best_val_r},
        )
        tb_writer.flush()
        tb_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hidden-dims", default="256,64")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tensorboard", action="store_true", help="Отключить логирование в TensorBoard")
    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x)

    train(
        features_dir=args.features_dir,
        out_dir=args.out_dir,
        hidden_dims=hidden_dims,
        dropout_rate=args.dropout_rate,
        aux_weight=args.aux_weight,
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
