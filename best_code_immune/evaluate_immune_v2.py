"""Оценка `EffectHead` v2 (fdr-взвешивание + direction aux) на иммунных ASE-данных.

Отличия от `evaluate_immune.py` — метрики выбраны так, чтобы быть напрямую
сопоставимыми с курсовой (раздел 5.1):
- Основная классификационная метрика — **AUC direction (over/under по знаку
  comb_es)**, а не "значим ASE или нет" (эта задача непредсказуема по
  последовательности, т.к. зависит от глубины покрытия).
  Считается на всех строках и отдельно на подмножестве fdr_comb_pval<0.05
  (наиболее надёжные ASE-вызовы, аналог фильтра "покрытие >= 10 ридов" из
  курсовой).
- Pearson r также считается дополнительно на подмножестве fdr<0.05.

Запуск:
    python evaluate_immune_v2.py --features-dir features_immune \
        --checkpoint runs/effect_head_immune_v2/best.pkl --plots-dir analysis_immune_v2
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from heads import forward
from train_immune import apply_scaler, build_input, load_immune_split, pearson_r, spearman_r


def direction_auc_curve(pred: np.ndarray, comb_es: np.ndarray):
    y = (comb_es > 0).astype(int)  # 1=under, 0=over (конвенция курсовой)
    auc = roc_auc_score(y, pred)
    flip = auc < 0.5
    score = -pred if flip else pred
    auc = max(auc, 1 - auc)
    fpr, tpr, _ = roc_curve(y, score)
    return auc, fpr, tpr


def per_cell_type_table(ct_names, pred, target, fdr, min_n=10) -> pd.DataFrame:
    rows = []
    for ct in sorted(set(ct_names.tolist())):
        mask = ct_names == ct
        if mask.sum() < min_n:
            continue
        sub_pred, sub_target, sub_fdr = pred[mask], target[mask], fdr[mask]
        y = (sub_target > 0).astype(int)
        try:
            auc = roc_auc_score(y, sub_pred)
            auc = max(auc, 1 - auc)
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "cell_type": ct,
                "n": int(mask.sum()),
                "n_fdr<0.05": int((sub_fdr < 0.05).sum()),
                "pearson_r": pearson_r(sub_pred, sub_target),
                "direction_auc": auc,
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def predict(params, config, mean, std, num_cell_types, x_lfc, cell_type_idx):
    x = build_input(apply_scaler(x_lfc, mean, std), cell_type_idx, num_cell_types)
    return np.asarray(forward(params, jnp.asarray(x), config, train=False)["z"])


def evaluate(features_dir: str, checkpoint_path: str, plots_dir: str | None) -> None:
    features_dir = Path(features_dir)
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]
    cell_type_vocab = ckpt["cell_type_vocab"]
    num_cell_types = ckpt["num_cell_types"]

    def ct_names_of(idx):
        return np.asarray(cell_type_vocab + ["unknown"])[np.clip(idx, 0, len(cell_type_vocab))]

    all_data = load_immune_split(features_dir, "train_all")
    chrom_str = np.asarray([c.decode() if isinstance(c, bytes) else str(c) for c in all_data["chrom"]])
    val_mask = chrom_str == "chr14"
    val_pred = predict(params, config, mean, std, num_cell_types, all_data["x_lfc"][val_mask], all_data["cell_type_idx"][val_mask])
    val_z = all_data["comb_es"][val_mask]
    val_fdr = all_data["fdr_comb_pval"][val_mask]

    print("=== Валидация (chr14) ===")
    print(f"Pearson r (все)      = {pearson_r(val_pred, val_z):.4f}  n={val_mask.sum()}")
    print(f"Pearson r (fdr<0.05) = {pearson_r(val_pred[val_fdr<0.05], val_z[val_fdr<0.05]):.4f}  n={(val_fdr<0.05).sum()}")

    test_path = features_dir / "test_features.npz"
    if not test_path.exists():
        print(f"[!] {test_path} не найден.")
        return

    test_data = load_immune_split(features_dir, "test")
    test_pred = predict(params, config, mean, std, num_cell_types, test_data["x_lfc"], test_data["cell_type_idx"])
    test_z = test_data["comb_es"]
    test_fdr = test_data["fdr_comb_pval"]
    sig_mask = test_fdr < 0.05

    r_all = pearson_r(test_pred, test_z)
    r_sig = pearson_r(test_pred[sig_mask], test_z[sig_mask])
    auc_all, fpr_all, tpr_all = direction_auc_curve(test_pred, test_z)
    auc_sig, fpr_sig, tpr_sig = direction_auc_curve(test_pred[sig_mask], test_z[sig_mask])

    print("\n=== Held-out тест (chr_test.h5) ===")
    print(f"Pearson r (все)         = {r_all:.4f}  n={len(test_z)}")
    print(f"Pearson r (fdr<0.05)    = {r_sig:.4f}  n={sig_mask.sum()}")
    print(f"AUC direction (все)     = {auc_all:.4f}")
    print(f"AUC direction (fdr<0.05)= {auc_sig:.4f}  <-- сравнимо с RF-на-эмбеддингах из курсовой (0.763)")

    ct_names_test = ct_names_of(test_data["cell_type_idx"])
    table = per_cell_type_table(ct_names_test, test_pred, test_z, test_fdr)
    print("\n=== По типу иммунной клетки (тест, n>=10) ===")
    print(table.to_string(index=False))

    if plots_dir is not None:
        _make_plots(Path(plots_dir), val_pred, val_z, test_pred, test_z, sig_mask, table, fpr_sig, tpr_sig, auc_sig, fpr_all, tpr_all, auc_all)


def _make_plots(plots_dir, val_pred, val_z, test_pred, test_z, sig_mask, table, fpr_sig, tpr_sig, auc_sig, fpr_all, tpr_all, auc_all):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    r = pearson_r(val_pred, val_z)
    plt.figure(figsize=(6, 6))
    plt.scatter(val_z, val_pred, s=4, alpha=0.25, color="steelblue")
    lims = [float(np.nanpercentile(val_z, 1)), float(np.nanpercentile(val_z, 99))]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Истинный comb_es")
    plt.ylabel("Предсказанный comb_es")
    plt.title(f"Валидация (chr14), v2: Pearson r={r:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "val_scatter_immune_v2.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(fpr_all, tpr_all, lw=2, label=f"все строки, AUC={auc_all:.3f}")
    plt.plot(fpr_sig, tpr_sig, lw=2, label=f"fdr<0.05, AUC={auc_sig:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Direction (over/under): ROC на chr_test.h5, v2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "test_roc_direction_v2.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, max(4, 0.3 * len(table))))
    plt.barh(table["cell_type"], table["pearson_r"], color="steelblue")
    plt.xlabel("Pearson r (тест)")
    plt.title("EffectHead v2: Pearson r по типу иммунной клетки")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(plots_dir / "test_pearson_by_cell_type_v2.png", dpi=150)
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
