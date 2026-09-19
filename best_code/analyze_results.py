"""Диагностика обученной EffectHead: графики + сравнение с RF-бейзлайном.

Запуск:
    python analyze_results.py --features-dir features --checkpoint runs/effect_head_v1/best.pkl --out-dir analysis
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve

from heads import forward
from train import apply_scaler, load_split

LABEL_MAP = {"none": 0, "over": 1, "under": 2}
COLORS = {"none": "green", "over": "orange", "under": "steelblue"}


def auc_pair(y_true, y_score, pos, neg=None):
    if neg is None:
        y_bin = (y_true == pos).astype(int)
        y_sc = y_score
    else:
        mask = (y_true == pos) | (y_true == neg)
        y_bin = (y_true[mask] == pos).astype(int)
        y_sc = y_score[mask]
    auc = roc_auc_score(y_bin, y_sc)
    flipped = auc < 0.5
    return (auc if not flipped else 1 - auc), flipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]

    def predict(x_raw):
        x = apply_scaler(x_raw, mean, std)
        return np.asarray(forward(params, jnp.asarray(x), config, train=False)["z"])

    # --- Val: scatter predicted vs true z ---
    val_data = load_split(features_dir, "val")
    val_pred = predict(val_data["x"])
    mask = np.isfinite(val_data["z"])
    r, _ = stats.pearsonr(val_pred[mask], val_data["z"][mask])

    plt.figure(figsize=(6, 6))
    plt.scatter(val_data["z"][mask], val_pred[mask], s=6, alpha=0.3, color="steelblue")
    lims = [-3, 3]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Истинный z (эффект варианта)")
    plt.ylabel("Предсказанный z (EffectHead)")
    plt.title(f"Валидация: предсказание vs истина (Pearson r={r:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_scatter.png", dpi=150)
    plt.close()

    # --- Test: ROC curves + AUC ---
    test_data = load_split(features_dir, "test")
    test_pred = predict(test_data["x"])
    y_true = test_data["consequence"]
    valid = y_true >= 0
    y_true = y_true[valid]
    y_score = test_pred[valid]

    pairs = [("over", "none"), ("under", "none"), ("over", "under")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    aucs = {}
    for ax, (pos, neg) in zip(axes, pairs):
        pos_c, neg_c = LABEL_MAP[pos], LABEL_MAP[neg]
        auc, flipped = auc_pair(y_true, y_score, pos_c, neg_c)
        mask_pair = (y_true == pos_c) | (y_true == neg_c)
        y_bin = (y_true[mask_pair] == pos_c).astype(int)
        y_sc = y_score[mask_pair] * (-1 if flipped else 1)
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{pos} vs {neg}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        ax.grid(alpha=0.3)
        aucs[f"{pos}_vs_{neg}"] = auc
    plt.suptitle("EffectHead (новый пайплайн): ROC на tableS1A")
    plt.tight_layout()
    plt.savefig(out_dir / "test_roc.png", dpi=150)
    plt.close()

    # --- Test: KDE distribution by class ---
    plt.figure(figsize=(10, 5))
    for label, code in LABEL_MAP.items():
        subset = y_score[y_true == code]
        if len(subset) < 2:
            continue
        kde = gaussian_kde(subset)
        xs = np.linspace(y_score.min(), y_score.max(), 200)
        plt.plot(xs, kde(xs), color=COLORS[label], lw=2, label=f"{label} (n={len(subset)})")
    plt.title("EffectHead: распределение предсказанного эффекта по истинному классу (tableS1A)")
    plt.xlabel("Предсказанный ẑ")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "test_kde.png", dpi=150)
    plt.close()

    print("=== Итоговые метрики ===")
    print(f"Val Pearson r = {r:.4f}, n={mask.sum()}")
    for k, v in aucs.items():
        print(f"Test AUC {k} = {v:.4f}")
    print(f"\nГрафики сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
