"""Обучаемая голова (`EffectHead`) поверх вектора log-fold-change AlphaGenome.

Реализована на чистом JAX (без Haiku), поскольку она не взаимодействует с
внутренностями модели AlphaGenome напрямую — обучается на закэшированных
признаках из `extract_features.py`. Это небольшой MLP (по умолчанию
`in_dim -> 256 -> 64 -> 1`), обучаемый через `jax.value_and_grad` + `optax`.

Для Stage 3 (`finetune_backbone.py`), где голова обучается совместно с
частью настоящих весов AlphaGenome, используются те же функции
`init_params`/`forward`/`loss_fn` — только к параметрам головы добавляются
несколько тензоров из `model._params`.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class HeadConfig:
    in_dim: int
    hidden_dims: tuple[int, ...] = (256, 64)
    dropout_rate: float = 0.2
    predict_aux_p_over: bool = True


def _linear_init(key, in_dim: int, out_dim: int):
    w_key, _ = jax.random.split(key)
    scale = jnp.sqrt(2.0 / in_dim)
    w = jax.random.normal(w_key, (in_dim, out_dim)) * scale
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def init_params(key, config: HeadConfig):
    dims = [config.in_dim, *config.hidden_dims]
    keys = jax.random.split(key, len(dims) + 1)
    trunk = [
        _linear_init(keys[i], dims[i], dims[i + 1]) for i in range(len(dims) - 1)
    ]
    head_z = _linear_init(keys[-2], dims[-1], 1)
    params = {"trunk": trunk, "head_z": head_z}
    if config.predict_aux_p_over:
        params["head_p_over"] = _linear_init(keys[-1], dims[-1], 1)
    return params


def _dropout(x, rate: float, rng, train: bool):
    if not train or rate <= 0.0:
        return x
    keep = jax.random.bernoulli(rng, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def forward(params, x, config: HeadConfig, rng=None, train: bool = False):
    """Возвращает dict с `z` (регрессия) и опционально `p_over_logit`."""

    h = x
    for i, layer in enumerate(params["trunk"]):
        h = h @ layer["w"] + layer["b"]
        h = jax.nn.relu(h)
        if train and rng is not None:
            rng, sub = jax.random.split(rng)
            h = _dropout(h, config.dropout_rate, sub, train)

    z_pred = (h @ params["head_z"]["w"] + params["head_z"]["b"])[:, 0]
    out = {"z": z_pred}
    if config.predict_aux_p_over and "head_p_over" in params:
        p_over_logit = (h @ params["head_p_over"]["w"] + params["head_p_over"]["b"])[:, 0]
        out["p_over_logit"] = p_over_logit
    return out


def huber_loss(pred, target, delta: float = 1.0):
    diff = pred - target
    abs_diff = jnp.abs(diff)
    quadratic = 0.5 * diff**2
    linear = delta * (abs_diff - 0.5 * delta)
    return jnp.where(abs_diff <= delta, quadratic, linear)


def bce_with_logits(logits, targets):
    # Численно стабильная BCE, targets могут быть "мягкими" (0..1), т.к.
    # `p_over` в данных — это posterior-вероятность, а не жёсткая метка.
    return jnp.maximum(logits, 0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def compute_loss(
    params,
    batch,
    config: HeadConfig,
    rng=None,
    train: bool = False,
    aux_weight: float = 0.0,
    huber_delta: float = 1.0,
):
    preds = forward(params, batch["x"], config, rng=rng, train=train)

    z_mask = jnp.isfinite(batch["z"])
    z_target = jnp.where(z_mask, batch["z"], 0.0)
    z_loss_per_ex = huber_loss(preds["z"], z_target, delta=huber_delta)
    z_loss = jnp.sum(z_loss_per_ex * z_mask) / jnp.maximum(jnp.sum(z_mask), 1.0)

    total_loss = z_loss
    metrics = {"z_loss": z_loss}

    if aux_weight > 0.0 and "p_over_logit" in preds:
        p_mask = jnp.isfinite(batch["p_over"])
        p_target = jnp.where(p_mask, batch["p_over"], 0.5)
        aux_loss_per_ex = bce_with_logits(preds["p_over_logit"], p_target)
        aux_loss = jnp.sum(aux_loss_per_ex * p_mask) / jnp.maximum(jnp.sum(p_mask), 1.0)
        total_loss = total_loss + aux_weight * aux_loss
        metrics["aux_loss"] = aux_loss

    metrics["loss"] = total_loss
    return total_loss, metrics
