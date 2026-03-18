import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

def create_train_step(model, optimizer, head_name='promoter_effect_head'):
    @jax.jit
    def train_step(params, state, opt_state, ref_batch, alt_batch, targets,
                   organism_index, negative_strand_mask, strand_reindexing):
        def loss_fn(params):
            pred_ref = model._predict(params, state, ref_batch, organism_index,
                                       negative_strand_mask=negative_strand_mask,
                                       strand_reindexing=strand_reindexing)[head_name]
            pred_alt = model._predict(params, state, alt_batch, organism_index,
                                       negative_strand_mask=negative_strand_mask,
                                       strand_reindexing=strand_reindexing)[head_name]
            diff = pred_alt - pred_ref
            mse = jnp.mean((diff.squeeze(-1) - targets) ** 2)
            return mse
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return train_step

def train_epoch(model, train_loader, optimizer, opt_state, train_step, head_name):
    total_loss = 0.0
    num_batches = 0
    first_organism = list(model._metadata.keys())[0]
    strand_reindexing = jax.device_put(model._metadata[first_organism].strand_reindexing,
                                       model._device_context._device)

    for ref_batch, alt_batch, targets in tqdm(train_loader.generate(), desc='Training'):
        ref_batch = jnp.array(ref_batch)
        alt_batch = jnp.array(alt_batch)
        targets = jnp.array(targets)

        current_batch = ref_batch.shape[0]
        organism_index = jnp.zeros((current_batch,), dtype=jnp.int32)
        negative_strand_mask = jnp.zeros((current_batch,), dtype=jnp.bool_)

        model._params, opt_state, loss = train_step(
            model._params, model._state, opt_state,
            ref_batch, alt_batch, targets,
            organism_index, negative_strand_mask, strand_reindexing
        )
        total_loss += loss
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss, opt_state

def validate(model, val_loader, head_name):
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    first_organism = list(model._metadata.keys())[0]
    strand_reindexing = jax.device_put(model._metadata[first_organism].strand_reindexing,
                                       model._device_context._device)

    for ref_batch, alt_batch, targets in val_loader.generate():
        ref_batch = jnp.array(ref_batch)
        alt_batch = jnp.array(alt_batch)
        targets = jnp.array(targets)

        current_batch = ref_batch.shape[0]
        organism_index = jnp.zeros((current_batch,), dtype=jnp.int32)
        negative_strand_mask = jnp.zeros((current_batch,), dtype=jnp.bool_)

        pred_ref = model._predict(model._params, model._state, ref_batch, organism_index,
                                   negative_strand_mask=negative_strand_mask,
                                   strand_reindexing=strand_reindexing)[head_name]
        pred_alt = model._predict(model._params, model._state, alt_batch, organism_index,
                                   negative_strand_mask=negative_strand_mask,
                                   strand_reindexing=strand_reindexing)[head_name]
        diff = (pred_alt - pred_ref).squeeze(-1)
        mse = jnp.mean((diff - targets) ** 2)
        total_loss += mse
        all_preds.extend(diff)
        all_targets.extend(targets)
        num_batches += 1

    avg_loss = total_loss / num_batches
    if len(all_preds) > 1:
        corr = jnp.corrcoef(jnp.array(all_preds), jnp.array(all_targets))[0, 1]
    else:
        corr = float('nan')
    return avg_loss, corr
