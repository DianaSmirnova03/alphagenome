"""Обучение `EffectHead` на иммунных ASE-данных (`extract_features_immune.py`).

Отличия от `train.py` (для SNP -> z-score экспрессии гена):

1. **Признак cell_type добавлен явно.** Один и тот же вариант (LFC-вектор от
   AlphaGenome) в этих данных сопровождается РАЗНЫМИ таргетами `comb_es` в
   зависимости от того, в каком типе иммунной клетки измерялась allele-
   specific экспрессия (37 типов: T/B/NK/моноциты/DC и подтипы). Старые
   попытки (`AG/905_finetune_full_immune.py`, `AG/1305_new_finetune_all_immune.py`)
   **не учитывали cell_type вообще** — то есть одному и тому же X (признаку
   варианта) сопоставлялось до 37 разных Y без какой-либо информации, по
   какому из них учиться сейчас. Это делает задачу нерешаемой (irreducible
   noise) и хорошо объясняет, почему обе попытки дали `Val corr: nan`
   (см. `AG/905_finetune_full_immune.log`, `AG/2_905_finetune_full_immune.log`).
   Здесь `cell_type` кодируется one-hot (37+1 "unknown" размерностей) и
   конкатенируется с LFC-вектором перед подачей в `EffectHead` — так голова
   может научиться разным весам по трекам для разных иммунных клеток.
2. **Валидация — по хромосоме, не по случайным строкам.** `--val-chrom 14`
   (по умолчанию) откладывает целую хромосому из `train_all_features.npz`,
   чтобы избежать утечки (варианты на соседних позициях одной хромосомы не
   попадают одновременно в train и val).
3. **Второй таргет `fdr_comb_pval` -> `is_sig` (FDR<0.05, ~6.5% строк).**
   Переиспользует ту же вспомогательную BCE-голову `head_p_over` из
   `heads.py` (мы просто передаём в неё `is_sig` вместо `p_over` — слот тот
   же, семантика — "значим ли ASE-эффект статистически").
4. **TensorBoard: расширенное логирование** — не только loss/Pearson/lr, но
   и норма градиента, отдельно Spearman r, гистограмма предсказаний и Pearson
   r по каждому из наиболее частых cell_type в валидации (чтобы видеть, для
   каких иммунных клеток модель работает лучше/хуже).

Запуск (на одном свободном GPU-ядре):
    CUDA_VISIBLE_DEVICES=0 python train_immune.py \
        --features-dir features_immune --out-dir runs/effect_head_immune_v1
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

from common import normalize_chrom, set_seed
from heads import HeadConfig, compute_loss, init_params, forward


def load_immune_split(features_dir: Path, split: str) -> dict:
    data = np.load(features_dir / f"{split}_features.npz", allow_pickle=True)
    return {
        "x_lfc": data["X"].astype(np.float32),
        "cell_type_idx": data["cell_type_idx"].astype(np.int64),
        "comb_es": data["comb_es"].astype(np.float32),
        "is_sig": data["is_sig"].astype(np.float32),
        "fdr_comb_pval": data["fdr_comb_pval"].astype(np.float32),
        "chrom": data["chrom"],
        "variant_id": data["variant_id"],
    }


def one_hot(idx: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(idx), num_classes), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


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


def spearman_r(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(target)
    if mask.sum() < 2:
        return float("nan")
    r, _ = stats.spearmanr(pred[mask], target[mask])
    return float(r)


def make_batches(n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def build_input(x_lfc_scaled: np.ndarray, cell_type_idx: np.ndarray, num_cell_types: int) -> np.ndarray:
    return np.concatenate([x_lfc_scaled, one_hot(cell_type_idx, num_cell_types)], axis=1)


def train(
    features_dir: str,
    out_dir: str,
    cell_type_vocab_path: str,
    val_chrom: str = "14",
    hidden_dims=(256, 64),
    dropout_rate: float = 0.2,
    aux_weight: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    tensorboard: bool = True,
):
    import json

    devices = jax.devices()
    print(f"JAX видит устройства: {devices}")

    set_seed(seed)
    features_dir = Path(features_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_type_vocab = json.loads(Path(cell_type_vocab_path).read_text())
    num_cell_types = len(cell_type_vocab) + 1  # +1 запасной "unknown"
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
    val_chrom_norm = normalize_chrom(val_chrom)  # "14" -> "chr14" (в фичах хромосома хранится с префиксом)
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
    p_over_train = all_data["is_sig"][train_mask]  # переиспользуем слот "p_over" под "is_sig"
    p_over_val = all_data["is_sig"][val_mask]

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
    def predict_z(params, x):
        return forward(params, x, config, train=False)["z"]

    rng_np = np.random.default_rng(seed)
    best_val_r = -np.inf
    best_state = None
    epochs_without_improvement = 0

    x_train_j = jnp.asarray(x_train)
    z_train_j = jnp.asarray(z_train)
    p_over_train_j = jnp.asarray(p_over_train)
    x_val_j = jnp.asarray(x_val)

    # Топ-10 самых частых cell_type в валидации — для разбивки Pearson по клеткам в TensorBoard
    ct_names_val = np.asarray(cell_type_vocab + ["unknown"])[np.clip(ct_val, 0, len(cell_type_vocab))]
    top_cell_types = pd_value_counts_top(ct_names_val, k=10)

    for epoch in range(max_epochs):
        epoch_losses, epoch_z_losses, epoch_aux_losses, epoch_grad_norms = [], [], [], []
        for batch_idx in make_batches(len(x_train), batch_size, rng_np):
            key, step_key = jax.random.split(key)
            batch = {
                "x": x_train_j[batch_idx],
                "z": z_train_j[batch_idx],
                "p_over": p_over_train_j[batch_idx],
            }
            params, opt_state, metrics = train_step(params, opt_state, batch, step_key)
            epoch_losses.append(float(metrics["loss"]))
            epoch_z_losses.append(float(metrics["z_loss"]))
            if "aux_loss" in metrics:
                epoch_aux_losses.append(float(metrics["aux_loss"]))
            epoch_grad_norms.append(float(metrics["grad_norm"]))

        val_pred = np.asarray(predict_z(params, x_val_j))
        val_r = pearson_r(val_pred, z_val)
        val_rho = spearman_r(val_pred, z_val)
        val_mse = float(np.nanmean((val_pred - z_val) ** 2))

        print(
            f"epoch {epoch:03d} | train_loss={np.mean(epoch_losses):.4f} |"
            f" val_pearson_r={val_r:.4f} | val_spearman_r={val_rho:.4f} | val_mse={val_mse:.4f}"
        )

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", float(np.mean(epoch_losses)), epoch)
            tb_writer.add_scalar("train/z_loss", float(np.mean(epoch_z_losses)), epoch)
            if epoch_aux_losses:
                tb_writer.add_scalar("train/aux_loss_is_sig", float(np.mean(epoch_aux_losses)), epoch)
            tb_writer.add_scalar("train/grad_norm", float(np.mean(epoch_grad_norms)), epoch)
            tb_writer.add_scalar("train/lr", float(schedule(epoch * steps_per_epoch)), epoch)
            tb_writer.add_scalar("val/pearson_r", float(val_r), epoch)
            tb_writer.add_scalar("val/spearman_r", float(val_rho), epoch)
            tb_writer.add_scalar("val/mse", float(val_mse), epoch)
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
                "val_chrom": str(val_chrom),
            },
            {"best_val_pearson_r": best_val_r},
        )
        tb_writer.flush()
        tb_writer.close()


def pd_value_counts_top(arr: np.ndarray, k: int) -> list[str]:
    import pandas as pd

    return pd.Series(arr).value_counts().head(k).index.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cell-type-vocab", default=None, help="По умолчанию <features-dir>/cell_type_vocab.json")
    parser.add_argument("--val-chrom", default="14")
    parser.add_argument("--hidden-dims", default="256,64")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
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
