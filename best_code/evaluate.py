"""Оценка обученной `EffectHead`: регрессия на валидации + AUC на tableS1A.

Метрики AUC (over/none, under/none, over/under) считаются той же формулой,
что в `AG/rf.py::auc_pair` — прямое сравнение с рабочим RF-бейзлайном
(AUC 0.78–0.90) на одних и тех же данных `tableS1A.tsv`.

Запуск:
    python evaluate.py --features-dir features --checkpoint runs/effect_head_v1/best.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from train import apply_scaler, load_split
from heads import forward


LABEL_MAP = {"none": 0, "over": 1, "under": 2}


def auc_pair(y_true: np.ndarray, y_score: np.ndarray, pos: int, neg: int | None):
    if neg is None:
        y_bin = (y_true == pos).astype(int)
        y_sc = y_score
    else:
        mask = (y_true == pos) | (y_true == neg)
        y_bin = (y_true[mask] == pos).astype(int)
        y_sc = y_score[mask]
    auc = roc_auc_score(y_bin, y_sc)
    return auc if auc >= 0.5 else 1 - auc


def evaluate(features_dir: str, checkpoint_path: str) -> None:
    features_dir = Path(features_dir)
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]

    def predict(x_raw: np.ndarray) -> np.ndarray:
        x = apply_scaler(x_raw, mean, std)
        preds = forward(params, jnp.asarray(x), config, train=False)
        return np.asarray(preds["z"])

    val_data = load_split(features_dir, "val")
    val_pred = predict(val_data["x"])
    mask = np.isfinite(val_data["z"])
    r, _ = stats.pearsonr(val_pred[mask], val_data["z"][mask])
    rho, _ = stats.spearmanr(val_pred[mask], val_data["z"][mask])
    mse = float(np.mean((val_pred[mask] - val_data["z"][mask]) ** 2))
    print("=== Валидация (регрессия z) ===")
    print(f"Pearson r  = {r:.4f}")
    print(f"Spearman r = {rho:.4f}")
    print(f"MSE        = {mse:.4f}")

    test_path = features_dir / "test_features.npz"
    if not test_path.exists():
        print(f"\n[!] {test_path} не найден — пропускаю оценку AUC на tableS1A.")
        return

    test_data = load_split(features_dir, "test")
    test_pred = predict(test_data["x"])
    y_true = test_data["consequence"]
    valid = y_true >= 0
    y_true = y_true[valid]
    y_score = test_pred[valid]

    auc_over_none = auc_pair(y_true, y_score, LABEL_MAP["over"], LABEL_MAP["none"])
    auc_under_none = auc_pair(y_true, y_score, LABEL_MAP["under"], LABEL_MAP["none"])
    auc_over_under = auc_pair(y_true, y_score, LABEL_MAP["over"], LABEL_MAP["under"])

    print("\n=== tableS1A (сравнение с RF-бейзлайном из AG/rf.py) ===")
    print(f"over vs none  AUC = {auc_over_none:.4f}")
    print(f"under vs none AUC = {auc_under_none:.4f}")
    print(f"over vs under AUC = {auc_over_under:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.features_dir, args.checkpoint)


if __name__ == "__main__":
    main()
