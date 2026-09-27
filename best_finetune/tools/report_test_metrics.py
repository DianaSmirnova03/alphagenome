"""Сводные метрики val/test + доп. графики для README (Promoter + Immune v1/v2).

Запуск (из корня пайплайна, нужны features/ и runs/*/best.pkl):

    export JAX_PLATFORMS=cpu
    python report_test_metrics.py --out-dir results_eval
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from evaluate import LABEL_MAP, auc_pair
from evaluate_immune_v2 import direction_auc_curve, per_cell_type_table, predict as immune_predict
from heads import forward
from train import apply_scaler, load_split, pearson_r
from train_immune import build_input, load_immune_split, spearman_r


def _promoter_metrics(features_dir: Path, checkpoint: Path) -> dict:
    ckpt = pickle.load(open(checkpoint, "rb"))
    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]

    def predict(x_raw):
        x = apply_scaler(x_raw, mean, std)
        return np.asarray(forward(params, jnp.asarray(x), config, train=False)["z"])

    val = load_split(features_dir, "val")
    val_pred = predict(val["x"])
    vmask = np.isfinite(val["z"])
    val_r, _ = stats.pearsonr(val_pred[vmask], val["z"][vmask])
    val_rho, _ = stats.spearmanr(val_pred[vmask], val["z"][vmask])

    test = load_split(features_dir, "test")
    test_pred = predict(test["x"])
    y_true = test["consequence"]
    valid = y_true >= 0
    y_true = y_true[valid]
    y_score = test_pred[valid]

    aucs = {
        "over_vs_none": float(auc_pair(y_true, y_score, LABEL_MAP["over"], LABEL_MAP["none"])),
        "under_vs_none": float(auc_pair(y_true, y_score, LABEL_MAP["under"], LABEL_MAP["none"])),
        "over_vs_under": float(auc_pair(y_true, y_score, LABEL_MAP["over"], LABEL_MAP["under"])),
    }
    counts = {
        "none": int((y_true == LABEL_MAP["none"]).sum()),
        "over": int((y_true == LABEL_MAP["over"]).sum()),
        "under": int((y_true == LABEL_MAP["under"]).sum()),
    }
    return {
        "checkpoint": str(checkpoint),
        "val": {
            "pearson_r": float(val_r),
            "spearman_r": float(val_rho),
            "n": int(vmask.sum()),
        },
        "test": {
            "split": "tableS1A.tsv (consequence labels; z unavailable)",
            "n": int(len(y_true)),
            "class_counts": counts,
            "auc": aucs,
        },
        "_arrays": {"val_pred": val_pred, "val_z": val["z"], "test_pred": test_pred, "y_true": y_true, "y_score": y_score},
    }


def _immune_metrics(features_dir: Path, checkpoint: Path, version: str) -> dict:
    ckpt = pickle.load(open(checkpoint, "rb"))
    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]
    num_cell_types = ckpt["num_cell_types"]

    all_data = load_immune_split(features_dir, "train_all")
    chrom_str = np.asarray([c.decode() if isinstance(c, bytes) else str(c) for c in all_data["chrom"]])
    val_mask = chrom_str == "chr14"
    val_pred = immune_predict(
        params, config, mean, std, num_cell_types,
        all_data["x_lfc"][val_mask], all_data["cell_type_idx"][val_mask],
    )
    val_z = all_data["comb_es"][val_mask]
    val_fdr = all_data["fdr_comb_pval"][val_mask]

    test_data = load_immune_split(features_dir, "test")
    test_pred = immune_predict(
        params, config, mean, std, num_cell_types,
        test_data["x_lfc"], test_data["cell_type_idx"],
    )
    test_z = test_data["comb_es"]
    test_fdr = test_data["fdr_comb_pval"]
    sig = test_fdr < 0.05

    auc_all, _, _ = direction_auc_curve(test_pred, test_z)
    auc_sig, _, _ = direction_auc_curve(test_pred[sig], test_z[sig])

    out = {
        "version": version,
        "checkpoint": str(checkpoint),
        "val_chr14": {
            "pearson_r_all": float(pearson_r(val_pred, val_z)),
            "pearson_r_fdr_lt_0.05": float(pearson_r(val_pred[val_fdr < 0.05], val_z[val_fdr < 0.05])),
            "n_all": int(val_mask.sum()),
            "n_fdr_lt_0.05": int((val_fdr < 0.05).sum()),
        },
        "test_chr_test_h5": {
            "pearson_r_all": float(pearson_r(test_pred, test_z)),
            "spearman_r_all": float(spearman_r(test_pred, test_z)),
            "pearson_r_fdr_lt_0.05": float(pearson_r(test_pred[sig], test_z[sig])),
            "direction_auc_all": float(auc_all),
            "direction_auc_fdr_lt_0.05": float(auc_sig),
            "n_all": int(len(test_z)),
            "n_fdr_lt_0.05": int(sig.sum()),
        },
        "_arrays": {
            "val_pred": val_pred,
            "val_z": val_z,
            "test_pred": test_pred,
            "test_z": test_z,
            "test_fdr": test_fdr,
            "sig_mask": sig,
        },
    }
    return out


def _plot_promoter_extra(m: dict, out_dir: Path) -> None:
    val_pred, val_z = m["_arrays"]["val_pred"], m["_arrays"]["val_z"]
    mask = np.isfinite(val_z)
    resid = val_pred[mask] - val_z[mask]

    plt.figure(figsize=(6, 4))
    plt.scatter(val_z[mask], resid, s=6, alpha=0.25, color="steelblue")
    plt.axhline(0, color="k", lw=1, ls="--")
    plt.xlabel("Истинный z (val)")
    plt.ylabel("Остаток ẑ − z")
    plt.title("Promoter: остатки регрессии на валидации")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "promoter_val_residual.png", dpi=150)
    plt.close()

    y_true, y_score = m["_arrays"]["y_true"], m["_arrays"]["y_score"]
    labels = ["none", "over", "under"]
    data = [y_score[y_true == LABEL_MAP[l]] for l in labels]
    plt.figure(figsize=(8, 5))
    plt.violinplot(data, showmeans=True)
    plt.xticks([1, 2, 3], labels)
    plt.ylabel("Предсказанный ẑ")
    plt.title("Promoter test (tableS1A): ẑ по классу consequence")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "promoter_test_violin_by_class.png", dpi=150)
    plt.close()


def _plot_immune_extra(m: dict, out_dir: Path, tag: str) -> None:
    ar = m["_arrays"]
    test_pred, test_z, sig = ar["test_pred"], ar["test_z"], ar["sig_mask"]

    def scatter(pred, z, title, fname):
        plt.figure(figsize=(6, 6))
        plt.scatter(z, pred, s=3, alpha=0.2, color="steelblue")
        lims = np.nanpercentile(z, [1, 99])
        plt.plot(lims, lims, "k--", lw=1)
        plt.xlabel("Истинный comb_es")
        plt.ylabel("Предсказанный comb_es")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    r_all = pearson_r(test_pred, test_z)
    scatter(test_pred, test_z, f"Immune {tag} test (all): r={r_all:.3f}", f"immune_{tag}_test_scatter_all.png")
    if sig.sum() >= 2:
        r_sig = pearson_r(test_pred[sig], test_z[sig])
        scatter(
            test_pred[sig],
            test_z[sig],
            f"Immune {tag} test (fdr<0.05): r={r_sig:.3f}, n={sig.sum()}",
            f"immune_{tag}_test_scatter_fdr005.png",
        )

    metrics = [
        ("Val Pearson", m["val_chr14"]["pearson_r_all"]),
        ("Test Pearson", m["test_chr_test_h5"]["pearson_r_all"]),
        ("Test AUC dir (fdr<0.05)", m["test_chr_test_h5"]["direction_auc_fdr_lt_0.05"]),
    ]
    plt.figure(figsize=(6, 4))
    plt.bar([x[0] for x in metrics], [x[1] for x in metrics], color=["#4c72b0", "#55a868", "#c44e52"])
    plt.ylim(0, 1)
    plt.ylabel("Значение метрики")
    plt.title(f"Immune {tag}: val vs test (сводка)")
    plt.xticks(rotation=15, ha="right")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / f"immune_{tag}_metrics_summary.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--immune-features-dir", default="features_immune")
    parser.add_argument("--promoter-checkpoint", default="runs/effect_head_v1/best.pkl")
    parser.add_argument("--immune-v1-checkpoint", default="runs/effect_head_immune_v1/best.pkl")
    parser.add_argument("--immune-v2-checkpoint", default="runs/effect_head_immune_v2/best.pkl")
    parser.add_argument("--out-dir", default="results_eval")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"promoter": None, "immune_v1": None, "immune_v2": None}

    p_ckpt = root / args.promoter_checkpoint
    if p_ckpt.exists():
        pm = _promoter_metrics(root / args.features_dir, p_ckpt)
        _plot_promoter_extra(pm, out_dir)
        pm_clean = {k: v for k, v in pm.items() if k != "_arrays"}
        report["promoter"] = pm_clean
        print(json.dumps(pm_clean, indent=2, ensure_ascii=False))
    else:
        print(f"[skip] promoter checkpoint {p_ckpt}")

    for tag, ckpt_rel in [("v1", args.immune_v1_checkpoint), ("v2", args.immune_v2_checkpoint)]:
        ckpt = root / ckpt_rel
        if not ckpt.exists():
            print(f"[skip] immune {tag} {ckpt}")
            continue
        im = _immune_metrics(root / args.immune_features_dir, ckpt, tag)
        _plot_immune_extra(im, out_dir, tag)
        im_clean = {k: v for k, v in im.items() if k != "_arrays"}
        report[f"immune_{tag}"] = im_clean
        print(json.dumps(im_clean, indent=2, ensure_ascii=False))

    json_path = out_dir / "test_metrics.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {json_path} and plots in {out_dir}/")


if __name__ == "__main__":
    main()
