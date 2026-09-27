"""Дифференцируемый CenterMask / GeneMask LFC внутри шага обучения."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from alphagenome.models import dna_output, variant_scorers
from alphagenome_research.model.variant_scoring import center_mask as center_mask_lib
from alphagenome_research.model.variant_scoring import gene_mask as gene_mask_lib


def sanitize_lfc(lfc: jnp.ndarray, *, clip: float = 20.0) -> jnp.ndarray:
    lfc = jnp.nan_to_num(lfc, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.clip(lfc, -clip, clip)


def _align_spatial_mask(masks: jnp.ndarray, seq_len: int) -> jnp.ndarray:
    """Приводит маску [..., L, 1] к длине оси последовательности predictions."""

    L = masks.shape[-2]
    if L == seq_len:
        return masks
    if L > seq_len:
        start = (L - seq_len) // 2
        return masks[..., start : start + seq_len, :]
    pad_before = (seq_len - L) // 2
    pad_after = seq_len - L - pad_before
    pad_width = [(0, 0)] * (masks.ndim - 2) + [(pad_before, pad_after), (0, 0)]
    return jnp.pad(masks, pad_width)


def _nonnegative_rna(ref_rna: jnp.ndarray, alt_rna: jnp.ndarray):
    """RNA predictions перед log-агрегацией — только неотрицательные (избегаем log NaN)."""

    return jnp.maximum(ref_rna, 0.0), jnp.maximum(alt_rna, 0.0)


def _center_lfc_diff_log2_sum(
    ref: jnp.ndarray,
    alt: jnp.ndarray,
    masks: jnp.ndarray,
) -> jnp.ndarray:
    """Стабильный аналог DIFF_LOG2_SUM (градиенты как у GeneMaskLFC, без log2(<=0))."""

    ref_p = jnp.maximum(ref, 0.0)
    alt_p = jnp.maximum(alt, 0.0)
    alt_sum = jnp.sum(alt_p, axis=0, where=masks)
    ref_sum = jnp.sum(ref_p, axis=0, where=masks)
    return jnp.log(alt_sum + 1e-3) - jnp.log(ref_sum + 1e-3)


def compute_rna_lfc_center_mask(
    apply_fn,
    backbone_params: dict,
    state,
    ref_onehot: jnp.ndarray,
    alt_onehot: jnp.ndarray,
    center_mask: jnp.ndarray,
    valid_track_mask: jnp.ndarray,
    organism_index: int,
    *,
    aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
):
    """CenterMask LFC по RNA_SEQ (как `extract_features_immune.py`)."""

    b = ref_onehot.shape[0]
    org = jnp.full((b,), organism_index, dtype=jnp.int32)
    ref_out = apply_fn(backbone_params, state, ref_onehot, org)
    alt_out = apply_fn(backbone_params, state, alt_onehot, org)
    ref_rna = ref_out["rna_seq"]["predictions_1bp"][..., valid_track_mask].astype(jnp.float32)
    alt_rna = alt_out["rna_seq"]["predictions_1bp"][..., valid_track_mask].astype(jnp.float32)
    ref_rna, alt_rna = _nonnegative_rna(ref_rna, alt_rna)
    seq_len = ref_rna.shape[-2]
    masks = _align_spatial_mask(center_mask.astype(jnp.bool_), seq_len)

    lfc = jax.vmap(_center_lfc_diff_log2_sum)(ref_rna, alt_rna, masks)
    return sanitize_lfc(lfc)


def compute_rna_lfc_gene_mask(
    apply_fn,
    backbone_params: dict,
    state,
    ref_onehot: jnp.ndarray,
    alt_onehot: jnp.ndarray,
    gene_mask: jnp.ndarray,
    valid_track_mask: jnp.ndarray,
    organism_index: int,
    *,
    scorer_settings=None,
):
    """GeneMask LFC (PromoterAI / SNP pipeline, Stage 3)."""

    from alphagenome.models import variant_scorers as vs_mod

    if scorer_settings is None:
        scorer_settings = vs_mod.GeneMaskLFCScorer(requested_output=dna_output.OutputType.RNA_SEQ)

    b = ref_onehot.shape[0]
    org = jnp.full((b,), organism_index, dtype=jnp.int32)
    ref_out = apply_fn(backbone_params, state, ref_onehot, org)
    alt_out = apply_fn(backbone_params, state, alt_onehot, org)
    ref_rna = ref_out["rna_seq"]["predictions_1bp"][..., valid_track_mask].astype(jnp.float32)
    alt_rna = alt_out["rna_seq"]["predictions_1bp"][..., valid_track_mask].astype(jnp.float32)
    ref_rna, alt_rna = _nonnegative_rna(ref_rna, alt_rna)
    gm = gene_mask[..., None] if gene_mask.ndim == 2 else gene_mask
    seq_len = ref_rna.shape[-2]
    if gm.shape[-2] != seq_len:
        gm = _align_spatial_mask(gm.astype(jnp.float32), seq_len)

    scorer_fn = functools.partial(gene_mask_lib._score_gene_variant, settings=scorer_settings)
    lfc = jax.vmap(scorer_fn)(ref_rna, alt_rna, gm)
    return sanitize_lfc(lfc[:, 0, :])
