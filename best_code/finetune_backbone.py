"""Stage 3 (опционально, продвинутый режим): настоящее дообучение весов.

В отличие от Stage 1+2 (`extract_features.py` + `train.py`), где backbone
AlphaGenome остаётся полностью замороженным и обучается только небольшая
голова над предпосчитанными LFC-признаками, здесь дополнительно
размораживается небольшой набор реальных весов модели (например, последний
линейный слой RNA_SEQ-головы — см. `inspect_params.py`, чтобы узнать точное
имя) и обучается совместно с `EffectHead` через `jax.grad` напрямую через
`apply_fn` (дифференцируемый форвард из
`alphagenome_research.model.dna_model.create_model`, см. `common.py`).

Формула агрегации log-fold-change по маске гена — `_score_gene_variant`
(GENE_MASK_LFC) импортируется прямо из
`alphagenome_research.model.variant_scoring.gene_mask` — той же функции,
которую внутри себя вызывает `model.score_variant` в Stage 1. Здесь она
применяется не к уже посчитанным (недифференцируемым) numpy-предсказаниям,
а прямо к выходу дифференцируемого `apply_fn`, поэтому градиент течёт через
неё в размороженные веса backbone.

ВАЖНО:
  - Использовать только после того, как Stage 1+2 отработал и дал baseline.
  - LR должен быть маленьким (1e-5..1e-4) — это всё-таки предобученные веса.
  - Требует предварительного запуска `sequence_dataset.py` (кэш one-hot
    последовательностей + маски гена), см. README.
  - Не проверялось на реальных весах модели (в этой рабочей среде нет
    доступа к Kaggle/GPU) — перед полным прогоном стоит сначала прогнать на
    небольшом `--max-examples` и убедиться, что loss не NaN/не расходится.

Запуск:
    python inspect_params.py --pattern rna_seq
    python finetune_backbone.py --features-dir seq_cache \
        --unfreeze-pattern "<имя модуля из inspect_params.py>" \
        --lr 1e-5 --out-dir runs/backbone_ft_v1
"""

from __future__ import annotations

import argparse
import functools
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import stats

from common import (
    build_differentiable_apply_fn,
    get_rna_seq_valid_mask,
    load_alphagenome_model,
    set_seed,
)
from heads import HeadConfig, compute_loss, init_params


def load_sequence_split(cache_dir: Path, split: str):
    data = np.load(cache_dir / f"{split}_sequences.npz")
    return {
        "ref": data["ref_onehot"].astype(np.float32),
        "alt": data["alt_onehot"].astype(np.float32),
        "gene_mask": data["gene_mask"].astype(np.float32),
        "z": data["z"].astype(np.float32),
    }


def split_params(params: dict, pattern: str):
    trainable = {k: v for k, v in params.items() if pattern.lower() in k.lower()}
    frozen = {k: v for k, v in params.items() if pattern.lower() not in k.lower()}
    n_trainable = sum(int(np.prod(a.shape)) for m in trainable.values() for a in m.values())
    print(f"Размороженные модули ({len(trainable)}): {list(trainable.keys())}")
    print(f"Всего обучаемых параметров backbone: {n_trainable:,}")
    return trainable, frozen


def run(
    features_dir: str,
    out_dir: str,
    unfreeze_pattern: str,
    fasta_path: str,
    gtf_feather_path: str,
    model_version: str = "all_folds",
    lr: float = 1e-5,
    batch_size: int = 8,
    max_epochs: int = 20,
    seed: int = 42,
):
    from alphagenome.models import dna_output, variant_scorers
    from alphagenome.models.dna_model import Organism
    from alphagenome_research.model import dna_model as ag_dna_model
    from alphagenome_research.model.variant_scoring import gene_mask as gene_mask_lib

    set_seed(seed)
    cache_dir = Path(features_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_sequence_split(cache_dir, "train")
    val_data = load_sequence_split(cache_dir, "val")

    model = load_alphagenome_model(
        model_version=model_version, fasta_path=fasta_path, gtf_feather_path=gtf_feather_path
    )
    _, apply_fn, _ = build_differentiable_apply_fn(model)
    state = model._state  # noqa: SLF001 — см. common.py
    valid_mask = jnp.asarray(get_rna_seq_valid_mask(model))
    organism_index = ag_dna_model.convert_to_organism_index(Organism.HOMO_SAPIENS)

    scorer_settings = variant_scorers.GeneMaskLFCScorer(requested_output=dna_output.OutputType.RNA_SEQ)
    scorer_fn = functools.partial(gene_mask_lib._score_gene_variant, settings=scorer_settings)
    batched_scorer = jax.vmap(scorer_fn, in_axes=(0, 0, 0))

    trainable_backbone, frozen_backbone = split_params(dict(model._params), unfreeze_pattern)

    head_config = HeadConfig(in_dim=int(valid_mask.sum()), predict_aux_p_over=False)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    head_params = init_params(init_key, head_config)

    params = {"backbone": trainable_backbone, "head": head_params}
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(params)

    def compute_lfc(full_backbone_params, batch):
        b = batch["ref"].shape[0]
        organism_idx_batch = jnp.full((b,), organism_index, dtype=jnp.int32)
        ref_out = apply_fn(full_backbone_params, state, batch["ref"], organism_idx_batch)
        alt_out = apply_fn(full_backbone_params, state, batch["alt"], organism_idx_batch)
        ref_rna = ref_out["rna_seq"]["predictions_1bp"][..., valid_mask].astype(jnp.float32)
        alt_rna = alt_out["rna_seq"]["predictions_1bp"][..., valid_mask].astype(jnp.float32)
        gene_mask = batch["gene_mask"][..., None]  # [B, L, 1] -> G=1
        lfc = batched_scorer(ref_rna, alt_rna, gene_mask)  # [B, 1, T]
        return lfc[:, 0, :]

    def loss_fn(params, batch, rng):
        full_backbone_params = {**frozen_backbone, **params["backbone"]}
        lfc = compute_lfc(full_backbone_params, batch)
        _, metrics = compute_loss(params["head"], {"x": lfc, "z": batch["z"], "p_over": jnp.full_like(batch["z"], jnp.nan)}, head_config, rng=rng, train=True)
        return metrics["loss"], metrics

    @jax.jit
    def train_step(params, opt_state, batch, rng):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def predict_z(params, batch):
        from heads import forward

        full_backbone_params = {**frozen_backbone, **params["backbone"]}
        lfc = compute_lfc(full_backbone_params, batch)
        return forward(params["head"], lfc, head_config, train=False)["z"]

    n_train = len(train_data["z"])
    rng_np = np.random.default_rng(seed)
    best_val_r, best_state = -np.inf, None

    for epoch in range(max_epochs):
        idx_all = rng_np.permutation(n_train)
        losses = []
        for start in range(0, n_train, batch_size):
            idx = idx_all[start : start + batch_size]
            batch = {
                "ref": jnp.asarray(train_data["ref"][idx]),
                "alt": jnp.asarray(train_data["alt"][idx]),
                "gene_mask": jnp.asarray(train_data["gene_mask"][idx]),
                "z": jnp.asarray(train_data["z"][idx]),
            }
            key, step_key = jax.random.split(key)
            params, opt_state, metrics = train_step(params, opt_state, batch, step_key)
            losses.append(float(metrics["loss"]))

        val_batch = {
            "ref": jnp.asarray(val_data["ref"]),
            "alt": jnp.asarray(val_data["alt"]),
            "gene_mask": jnp.asarray(val_data["gene_mask"]),
            "z": jnp.asarray(val_data["z"]),
        }
        val_pred = np.asarray(predict_z(params, val_batch))
        mask = np.isfinite(val_data["z"])
        val_r = float(stats.pearsonr(val_pred[mask], val_data["z"][mask])[0])
        print(f"epoch {epoch:03d} | train_loss={np.mean(losses):.4f} | val_pearson_r={val_r:.4f}")

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = jax.tree_util.tree_map(lambda a: np.array(a), params)

    with open(out_dir / "best.pkl", "wb") as f:
        pickle.dump(
            {
                "params": best_state,
                "head_config": head_config,
                "unfreeze_pattern": unfreeze_pattern,
                "best_val_pearson_r": best_val_r,
            },
            f,
        )
    print(f"Сохранено {out_dir / 'best.pkl'} (best val_pearson_r={best_val_r:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, help="Папка с *_sequences.npz из sequence_dataset.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--unfreeze-pattern", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--gtf-feather", required=True)
    parser.add_argument("--model-version", default="all_folds")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        features_dir=args.features_dir,
        out_dir=args.out_dir,
        unfreeze_pattern=args.unfreeze_pattern,
        fasta_path=args.fasta,
        gtf_feather_path=args.gtf_feather,
        model_version=args.model_version,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
