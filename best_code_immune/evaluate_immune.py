"""Оценка `EffectHead`, обученной на иммунных ASE-данных.

Метрики:
- Pearson/Spearman r и MSE на валидации (отложенная хромосома) и на
  held-out `chr_test.h5` — регрессия `comb_es`.
- AUC для бинарной задачи "значим ли ASE-эффект" (`fdr_comb_pval < 0.05`,
  ~6.5% строк) — аналог AUC over/under/none из `evaluate.py`, но здесь
  естественная бинарная метка вместо 3 классов.
- Разбивка Pearson r по каждому cell_type на тесте (таблица + график) —
  показывает, для каких иммунных клеток модель предсказывает лучше/хуже.

Запуск:
    python evaluate_immune.py --features-dir features_immune \
        --checkpoint runs/effect_head_immune_v1/best.pkl --plots-dir analysis_immune
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from heads import forward
from train_immune import apply_scaler, build_input, load_immune_split, pearson_r, spearman_r


def predict(params, config, mean, std, num_cell_types, x_lfc: np.ndarray, cell_type_idx: np.ndarray) -> np.ndarray:
    x_lfc_scaled = apply_scaler(x_lfc, mean, std)
    x = build_input(x_lfc_scaled, cell_type_idx, num_cell_types)
    preds = forward(params, jnp.asarray(x), config, train=False)
    return np.asarray(preds["z"])


def per_cell_type_table(ct_names: np.ndarray, pred: np.ndarray, target: np.ndarray, min_n: int = 10) -> pd.DataFrame:
    rows = []
    for ct in sorted(set(ct_names.tolist())):
        mask = ct_names == ct
        if mask.sum() < min_n:
            continue
        rows.append(
            {
                "cell_type": ct,
                "n": int(mask.sum()),
                "pearson_r": pearson_r(pred[mask], target[mask]),
                "mse": float(np.nanmean((pred[mask] - target[mask]) ** 2)),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def evaluate(features_dir: str, checkpoint_path: str, plots_dir: str | None) -> None:
    features_dir = Path(features_dir)
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]
    cell_type_vocab = ckpt["cell_type_vocab"]
    num_cell_types = ckpt["num_cell_types"]

    def ct_names_of(cell_type_idx: np.ndarray) -> np.ndarray:
        return np.asarray(cell_type_vocab + ["unknown"])[np.clip(cell_type_idx, 0, len(cell_type_vocab))]

    # --- Валидация (отложенная хромосома из train_all) ---
    all_data = load_immune_split(features_dir, "train_all")
    chrom_str = np.asarray([c.decode() if isinstance(c, bytes) else str(c) for c in all_data["chrom"]])
    val_mask = chrom_str == "chr14"
    val_pred = predict(params, config, mean, std, num_cell_types, all_data["x_lfc"][val_mask], all_data["cell_type_idx"][val_mask])
    val_z = all_data["comb_es"][val_mask]
    print("=== Валидация (отложенная хромосома 14) ===")
    print(f"Pearson r  = {pearson_r(val_pred, val_z):.4f}")
    print(f"Spearman r = {spearman_r(val_pred, val_z):.4f}")
    print(f"MSE        = {float(np.nanmean((val_pred - val_z) ** 2)):.4f}")
    print(f"n = {val_mask.sum()}")

    # --- Тест (chr_test.h5, полностью held-out) ---
    test_path = features_dir / "test_features.npz"
    if not test_path.exists():
        print(f"\n[!] {test_path} не найден — пропускаю оценку на held-out тесте.")
        return

    test_data = load_immune_split(features_dir, "test")
    test_pred = predict(params, config, mean, std, num_cell_types, test_data["x_lfc"], test_data["cell_type_idx"])
    test_z = test_data["comb_es"]
    test_fdr = test_data["fdr_comb_pval"]
    is_sig = (test_fdr < 0.05).astype(int)

    r = pearson_r(test_pred, test_z)
    rho = spearman_r(test_pred, test_z)
    mse = float(np.nanmean((test_pred - test_z) ** 2))
    print("\n=== Held-out тест (chr_test.h5) ===")
    print(f"Pearson r  = {r:.4f}")
    print(f"Spearman r = {rho:.4f}")
    print(f"MSE        = {mse:.4f}")
    print(f"n = {len(test_z)}")

    auc = roc_auc_score(is_sig, np.abs(test_pred))
    print(f"\nAUC (значим ASE, FDR<0.05, по |predicted comb_es|) = {auc:.4f}")
    print(f"Доля значимых (FDR<0.05) в тесте: {is_sig.mean():.4f}")

    ct_names_test = ct_names_of(test_data["cell_type_idx"])
    table = per_cell_type_table(ct_names_test, test_pred, test_z)
    print("\n=== Pearson r по типу иммунной клетки (тест, n>=10) ===")
    print(table.to_string(index=False))

    if plots_dir is not None:
        _make_plots(Path(plots_dir), val_pred, val_z, test_pred, test_z, is_sig, table)


def _make_plots(plots_dir: Path, val_pred, val_z, test_pred, test_z, is_sig, table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Scatter на валидации
    r = pearson_r(val_pred, val_z)
    plt.figure(figsize=(6, 6))
    plt.scatter(val_z, val_pred, s=4, alpha=0.25, color="steelblue")
    lims = [float(np.nanpercentile(val_z, 1)), float(np.nanpercentile(val_z, 99))]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Истинный comb_es")
    plt.ylabel("Предсказанный comb_es (EffectHead)")
    plt.title(f"Валидация (chr14): предсказание vs истина (Pearson r={r:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "val_scatter_immune.png", dpi=150)
    plt.close()

    # ROC для "значим ASE"
    fpr, tpr, _ = roc_curve(is_sig, np.abs(test_pred))
    auc = roc_auc_score(is_sig, np.abs(test_pred))
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Значим ASE (FDR<0.05) по |predicted comb_es|: ROC на chr_test.h5")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "test_roc_immune.png", dpi=150)
    plt.close()

    # Барчарт Pearson r по cell_type
    plt.figure(figsize=(8, max(4, 0.3 * len(table))))
    plt.barh(table["cell_type"], table["pearson_r"], color="steelblue")
    plt.xlabel("Pearson r (тест)")
    plt.title("EffectHead: Pearson r по типу иммунной клетки")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(plots_dir / "test_pearson_by_cell_type.png", dpi=150)
    plt.close()

    print(f"\nГрафики сохранены в {plots_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plots-dir", default=None)
    args = parser.parse_args()
    evaluate(args.features_dir, args.checkpoint, args.plots_dir)


if __name__ == "__main__":
    main()
