"""Утилиты батча для E2E immune (дедуп SNP в батче, безопасные градиенты)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def dedupe_batch_by_variant(
    row_indices: np.ndarray,
    chrom: np.ndarray,
    pos: np.ndarray,
    ref: np.ndarray,
    alt: np.ndarray,
):
    keys = []
    for i in row_indices:
        keys.append((chrom[i], int(pos[i]), ref[i], alt[i]))
    unique_keys = list(dict.fromkeys(keys))
    key_to_u = {k: ui for ui, k in enumerate(unique_keys)}
    u_for_row = np.asarray([key_to_u[k] for k in keys], dtype=np.int64)
    return unique_keys, u_for_row


def zero_nonfinite_grads(grads):
    return jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
