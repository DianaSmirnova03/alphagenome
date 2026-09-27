"""Figures for README: training curves + test metrics. Minimal labels (EN).

    export JAX_PLATFORMS=cpu
    python make_figures.py --task promoter --out-dir analysis
    python make_figures.py --task immune --out-dir analysis_v2
    python make_figures.py --task all --promoter-out analysis --immune-out analysis_v2
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
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from tensorboard.backend.event_processing import event_accumulator

from evaluate import LABEL_MAP, auc_pair
from evaluate_immune_v2 import direction_auc_curve, per_cell_type_table, predict as immune_predict
from heads import forward
from train import apply_scaler, load_split, pearson_r
from train_immune import load_immune_split, spearman_r

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

RF_AUC = {
    "over_vs_none": (0.73, 0.78),
    "under_vs_none": (0.75, 0.78),
    "over_vs_under": (0.86, 0.90),
}


def _tb_scalars(logdir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    ea = event_accumulator.EventAccumulator(
        str(logdir), size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    ev = ea.Scalars(tag)
    steps = np.array([e.step for e in ev], dtype=np.int64)
    vals = np.array([e.value for e in ev], dtype=np.float64)
    return steps, vals


def plot_training_promoter(logdir: Path, out_dir: Path) -> None:
    ep, loss = _tb_scalars(logdir, "train/loss")
    _, val_r = _tb_scalars(logdir, "val/pearson_r")
    n = min(len(ep), len(val_r))
    ep, loss, val_r = ep[:n], loss[:n], val_r[:n]
    best_i = int(np.argmax(val_r))

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(ep, loss, color="#1f77b4", lw=1.5, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(ep, val_r, color="#ff7f0e", lw=1.5, label="val Pearson r")
    ax2.scatter([ep[best_i]], [val_r[best_i]], c="red", s=28, zorder=5)
    ax2.set_ylabel("Pearson r")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "train_curve.png")
    plt.close(fig)


def plot_training_immune(logdir: Path, out_dir: Path) -> None:
    ep, loss = _tb_scalars(logdir, "train/loss")
    _, val_r = _tb_scalars(logdir, "val/pearson_r")
    n = min(len(ep), len(val_r))
    ep, loss, val_r = ep[:n], loss[:n], val_r[:n]
    best_i = int(np.argmax(val_r))

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(ep, loss, color="#1f77b4", lw=1.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(ep, val_r, color="#ff7f0e", lw=1.5)
    ax2.scatter([ep[best_i]], [val_r[best_i]], c="red", s=28, zorder=5)
    ax2.set_ylabel("Pearson r")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "train_curve.png")
    plt.close(fig)


def _promoter_predict(features_dir: Path, checkpoint: Path):
    ckpt = pickle.load(open(checkpoint, "rb"))
    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]

    def predict(x_raw):
        x = apply_scaler(x_raw, mean, std)
        return np.asarray(forward(params, jnp.asarray(x), config, train=False)["z"])

    return predict


def promoter_figures(features_dir: Path, checkpoint: Path, out_dir: Path, tb_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if tb_dir.exists():
        plot_training_promoter(tb_dir, out_dir)

    predict = _promoter_predict(features_dir, checkpoint)
    val = load_split(features_dir, "val")
    val_pred = predict(val["x"])
    vmask = np.isfinite(val["z"])
    r, _ = stats.pearsonr(val_pred[vmask], val["z"][vmask])

    plt.figure(figsize=(5, 5))
    plt.scatter(val["z"][vmask], val_pred[vmask], s=5, alpha=0.25, color="steelblue")
    lims = [-3, 3]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("z")
    plt.ylabel("pred")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_scatter.png")
    plt.close()

    resid = val_pred[vmask] - val["z"][vmask]
    plt.figure(figsize=(5, 3.5))
    plt.scatter(val["z"][vmask], resid, s=5, alpha=0.25, color="steelblue")
    plt.axhline(0, color="k", lw=1, ls="--")
    plt.xlabel("z")
    plt.ylabel("residual")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_residual.png")
    plt.close()

    test = load_split(features_dir, "test")
    test_pred = predict(test["x"])
    y_true = test["consequence"][test["consequence"] >= 0]
    y_score = test_pred[test["consequence"] >= 0]

    pairs = [("over", "none"), ("under", "none"), ("over", "under")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    aucs = {}
    for ax, (pos, neg) in zip(axes, pairs):
        pos_c, neg_c = LABEL_MAP[pos], LABEL_MAP[neg]
        mask = (y_true == pos_c) | (y_true == neg_c)
        y_bin = (y_true[mask] == pos_c).astype(int)
        sc = y_score[mask]
        auc_v = float(auc_pair(y_true, y_score, pos_c, neg_c))
        sc_plot = y_score[mask]
        if roc_auc_score(y_bin, sc_plot) < 0.5:
            sc_plot = -sc_plot
        fpr, tpr, _ = roc_curve(y_bin, sc_plot)
        ax.plot(fpr, tpr, lw=1.5, label=f"AUC={auc_v:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{pos} vs {neg}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        aucs[f"{pos}_vs_{neg}"] = auc_v
    fig.tight_layout()
    fig.savefig(out_dir / "test_roc.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (pos, neg) in zip(axes, pairs):
        pos_c, neg_c = LABEL_MAP[pos], LABEL_MAP[neg]
        mask = (y_true == pos_c) | (y_true == neg_c)
        y_bin = (y_true[mask] == pos_c).astype(int)
        sc = y_score[mask]
        auc_v = float(auc_pair(y_true, y_score, pos_c, neg_c))
        if auc_v < 0.5:
            sc = -sc
        prec, rec, _ = precision_recall_curve(y_bin, sc)
        ap = sk_auc(rec, prec)
        ax.plot(rec, prec, lw=1.5, label=f"AP={ap:.3f}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"{pos} vs {neg}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "test_pr.png")
    plt.close(fig)

    order = np.argsort(y_score)
    n = len(y_score)
    deciles = []
    frac_over = []
    for k in range(10):
        lo, hi = k * n // 10, (k + 1) * n // 10
        sl = order[lo:hi]
        deciles.append(float(np.mean(y_score[sl])))
        frac_over.append(float(np.mean(y_true[sl] == LABEL_MAP["over"])))
    plt.figure(figsize=(5, 3.5))
    plt.plot(deciles, frac_over, "o-", color="steelblue")
    plt.xlabel("pred decile mean")
    plt.ylabel("P(over)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "test_calibration_over.png")
    plt.close()

    plt.figure(figsize=(8, 3.5))
    for label, code in LABEL_MAP.items():
        subset = y_score[y_true == code]
        if len(subset) < 2:
            continue
        plt.hist(subset, bins=30, alpha=0.45, density=True, label=label)
    plt.xlabel("pred")
    plt.ylabel("density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "test_kde.png")
    plt.close()

    labels = ["over|none", "under|none", "over|under"]
    keys = ["over_vs_none", "under_vs_none", "over_vs_under"]
    eh = [aucs[k] for k in keys]
    rf_lo = [RF_AUC[k][0] for k in keys]
    rf_hi = [RF_AUC[k][1] for k in keys]
    rf_mid = [(a + b) / 2 for a, b in zip(rf_lo, rf_hi)]
    x = np.arange(3)
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - w / 2, rf_mid, w, yerr=[np.array(rf_mid) - rf_lo, np.array(rf_hi) - np.array(rf_mid)], capsize=3, label="RF", color="#aaa")
    plt.bar(x + w / 2, eh, w, label="EffectHead", color="steelblue")
    plt.xticks(x, labels)
    plt.ylabel("AUC")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "test_auc_rf_vs_head.png")
    plt.close()

    metrics = {
        "val": {"pearson_r": float(r), "n": int(vmask.sum())},
        "test": {"n": int(len(y_true)), "auc": aucs},
    }
    return metrics


def immune_figures(
    features_dir: Path, checkpoint: Path, out_dir: Path, tb_dir: Path, tag: str = "v2"
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if tb_dir.exists() and tag == "v2":
        plot_training_immune(tb_dir, out_dir)

    ckpt = pickle.load(open(checkpoint, "rb"))
    params, config = ckpt["params"], ckpt["config"]
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]
    num_cell_types = ckpt["num_cell_types"]
    cell_type_vocab = ckpt["cell_type_vocab"]

    all_data = load_immune_split(features_dir, "train_all")
    chrom_str = np.asarray([c.decode() if isinstance(c, bytes) else str(c) for c in all_data["chrom"]])
    val_mask = chrom_str == "chr14"
    val_pred = immune_predict(
        params, config, mean, std, num_cell_types,
        all_data["x_lfc"][val_mask], all_data["cell_type_idx"][val_mask],
    )
    val_z = all_data["comb_es"][val_mask]

    test_data = load_immune_split(features_dir, "test")
    test_pred = immune_predict(
        params, config, mean, std, num_cell_types,
        test_data["x_lfc"], test_data["cell_type_idx"],
    )
    test_z = test_data["comb_es"]
    test_fdr = test_data["fdr_comb_pval"]
    sig = test_fdr < 0.05

    r_val = pearson_r(val_pred, val_z)
    plt.figure(figsize=(5, 5))
    plt.scatter(val_z, val_pred, s=3, alpha=0.2, color="steelblue")
    lims = np.nanpercentile(val_z, [1, 99])
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("comb_es")
    plt.ylabel("pred")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"val_scatter_immune_{tag}.png" if tag != "v2" else out_dir / "val_scatter_immune_v2.png")
    plt.close()

    auc_all, fpr_a, tpr_a = direction_auc_curve(test_pred, test_z)
    auc_sig, fpr_s, tpr_s = direction_auc_curve(test_pred[sig], test_z[sig])

    plt.figure(figsize=(4, 4))
    plt.plot(fpr_a, tpr_a, lw=1.5, label=f"all AUC={auc_all:.3f}")
    plt.plot(fpr_s, tpr_s, lw=1.5, label=f"fdr<0.05 AUC={auc_sig:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "test_roc_direction_v2.png" if tag == "v2" else out_dir / f"test_roc_direction_{tag}.png")
    plt.close()

    y_dir = (test_z[sig] > 0).astype(int)
    sc = test_pred[sig]
    if roc_auc_score(y_dir, sc) < 0.5:
        sc = -sc
    prec, rec, _ = precision_recall_curve(y_dir, sc)
    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec, lw=1.5, label=f"AP={sk_auc(rec, prec):.3f}")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "test_pr_direction_fdr005.png")
    plt.close()

    ct_names = np.asarray(cell_type_vocab + ["unknown"])[
        np.clip(test_data["cell_type_idx"], 0, len(cell_type_vocab))
    ]
    table = per_cell_type_table(ct_names, test_pred, test_z, test_fdr, min_n=10)
    if len(table) > 0:
        plt.figure(figsize=(8, max(3, 0.22 * len(table))))
        plt.barh(table["cell_type"], table["pearson_r"], color="steelblue")
        plt.xlabel("Pearson r")
        plt.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(out_dir / "test_pearson_by_cell_type_v2.png" if tag == "v2" else out_dir / f"test_pearson_by_cell_type_{tag}.png")
        plt.close()

        pivot_r = table.set_index("cell_type")["pearson_r"]
        pivot_a = table.set_index("cell_type")["direction_auc"]
        mat = np.stack([pivot_r.values, pivot_a.values], axis=1)
        fig, ax = plt.subplots(figsize=(4, max(3, 0.25 * len(table))))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pearson r", "dir AUC"])
        ax.set_yticks(range(len(table)))
        ax.set_yticklabels(table["cell_type"])
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / "test_heatmap_cell_type.png")
        plt.close(fig)

    if sig.sum() >= 2:
        r_all = pearson_r(test_pred, test_z)
        plt.figure(figsize=(5, 5))
        plt.scatter(test_z, test_pred, s=3, alpha=0.15, color="steelblue")
        lims = np.nanpercentile(test_z, [1, 99])
        plt.plot(lims, lims, "k--", lw=1)
        plt.xlabel("comb_es")
        plt.ylabel("pred")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "test_scatter_all.png")
        plt.close()

        r_sig = pearson_r(test_pred[sig], test_z[sig])
        plt.figure(figsize=(5, 5))
        plt.scatter(test_z[sig], test_pred[sig], s=8, alpha=0.35, color="steelblue")
        lims = np.nanpercentile(test_z[sig], [1, 99])
        plt.plot(lims, lims, "k--", lw=1)
        plt.xlabel("comb_es")
        plt.ylabel("pred")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "test_scatter_fdr005.png")
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.hist(np.abs(test_z[sig]), bins=40, alpha=0.5, label="|comb_es|", density=True)
        plt.hist(np.abs(test_pred[sig]), bins=40, alpha=0.5, label="|pred|", density=True)
        plt.xlabel("magnitude")
        plt.ylabel("density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "test_hist_abs_fdr005.png")
        plt.close()

    return {
        "val_chr14": {"pearson_r": float(r_val), "n": int(val_mask.sum())},
        "test": {
            "pearson_r_all": float(pearson_r(test_pred, test_z)),
            "pearson_r_fdr_lt_0.05": float(pearson_r(test_pred[sig], test_z[sig])),
            "direction_auc_all": float(auc_all),
            "direction_auc_fdr_lt_0.05": float(auc_sig),
            "n_all": int(len(test_z)),
            "n_fdr_lt_0.05": int(sig.sum()),
        },
    }


def metrics_card(json_path: Path, out_path: Path) -> None:
    d = json.loads(json_path.read_text())
    lines = ["test metrics"]
    if d.get("promoter"):
        p = d["promoter"]
        lines.append(f"promoter val r={p['val']['pearson_r']:.3f}")
        for k, v in p["test"]["auc"].items():
            lines.append(f"  test AUC {k}: {v:.3f}")
    for key in ("immune_v1", "immune_v2"):
        block = d.get(key)
        if not block:
            continue
        t = block.get("test_chr_test_h5") or block.get("test")
        if not t:
            continue
        r = t.get("pearson_r_all") or t.get("pearson_r")
        da = t.get("direction_auc_fdr_lt_0.05")
        if da is not None:
            lines.append(f"{key} test r={r:.3f} dirAUC={da:.3f}")
        elif r is not None:
            lines.append(f"{key} test r={r:.3f}")
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(lines) + 0.5))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["promoter", "immune", "all"], default="all")
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--immune-features-dir", default="features_immune")
    parser.add_argument("--promoter-checkpoint", default="runs/effect_head_v1/best.pkl")
    parser.add_argument("--immune-v2-checkpoint", default="runs/effect_head_immune_v2/best.pkl")
    parser.add_argument("--promoter-tb", default="runs/effect_head_v1/tensorboard")
    parser.add_argument("--immune-tb", default="runs/effect_head_immune_v2/tensorboard")
    parser.add_argument("--promoter-out", default="analysis")
    parser.add_argument("--immune-out", default="analysis_v2")
    parser.add_argument("--metrics-json", default="results_eval/test_metrics.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    report = {}

    if args.task in ("promoter", "all"):
        report["promoter"] = promoter_figures(
            root / args.features_dir,
            root / args.promoter_checkpoint,
            root / args.promoter_out,
            root / args.promoter_tb,
        )

    if args.task in ("immune", "all"):
        report["immune_v2"] = immune_figures(
            root / args.immune_features_dir,
            root / args.immune_v2_checkpoint,
            root / args.immune_out,
            root / args.immune_tb,
            tag="v2",
        )

    json_path = root / args.metrics_json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if json_path.exists():
        existing = json.loads(json_path.read_text())
    else:
        existing = {}
    if report.get("promoter"):
        existing["promoter"] = {**existing.get("promoter", {}), **report["promoter"]}
    if report.get("immune_v2"):
        existing["immune_v2"] = {**existing.get("immune_v2", {}), **report["immune_v2"]}
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    metrics_card(json_path, root / args.promoter_out / "metrics_card.png")
    if args.task == "all":
        import shutil

        shutil.copy(root / args.promoter_out / "metrics_card.png", root / args.immune_out / "metrics_card.png")

    print(f"Wrote figures to {args.promoter_out} / {args.immune_out}")


if __name__ == "__main__":
    main()
