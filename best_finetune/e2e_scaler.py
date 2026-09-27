"""Scaler LFC по онлайн-forward (без npz X)."""

from __future__ import annotations

import numpy as np

from train import fit_scaler, make_batches


def fit_lfc_scaler_from_rows(
    row_indices: np.ndarray,
    *,
    batch_size: int,
    lfc_rows_fn,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """lfc_rows_fn(batch_row_indices) -> (B, D) сырой LFC."""

    rng = np.random.default_rng(seed)
    chunks = []
    for b in make_batches(len(row_indices), batch_size, rng, shuffle=False):
        bi = row_indices[b]
        chunks.append(np.asarray(lfc_rows_fn(bi), dtype=np.float32))
    if not chunks:
        raise RuntimeError("fit_lfc_scaler_from_rows: нет данных")
    return fit_scaler(np.concatenate(chunks, axis=0))
