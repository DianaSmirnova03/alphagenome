import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

def create_train_step(model, optimizer, head_name='promoter_effect_head'):
    organism_enum = next(iter(model._metadata.keys()))
    strand_reindexing = model._metadata[organism_enum].strand_reindexing
    strand_reindexing = jax.device_put(strand_reindexing, model._device_context._device)

    @jax.jit
    def train_step(params, state, opt_state, batch):
        ref = batch['ref_seq']
        alt = batch['alt_seq']
        targets = batch['targets']
        mask = batch['gene_mask']
        organism_index = jnp.zeros((ref.shape[0],), dtype=jnp.int32)
        neg_strand = jnp.zeros((ref.shape[0],), dtype=jnp.bool_)

        def loss_fn(params):
            pred_ref = model._predict(params, state, ref, organism_index,
                                       negative_strand_mask=neg_strand,
                                       strand_reindexing=strand_reindexing,
                                       gene_mask=mask)[head_name]
            pred_alt = model._predict(params, state, alt, organism_index,
                                       negative_strand_mask=neg_strand,
                                       strand_reindexing=strand_reindexing,
                                       gene_mask=mask)[head_name]
            diff = pred_alt - pred_ref
            mse = jnp.mean((diff.squeeze(-1) - targets) ** 2)
            return mse

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return train_step

def train_epoch(model, train_loader, optimizer, opt_state, train_step):
    total_loss = 0.0
    num_batches = 0
    for batch in tqdm(train_loader.generate(), desc='Training'):
        model._params, opt_state, loss = train_step(
            model._params, model._state, opt_state, batch
        )
        total_loss += loss
        num_batches += 1
    return total_loss / num_batches, opt_state

def validate(model, val_loader, head_name):
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    organism_enum = next(iter(model._metadata.keys()))
    strand_reindexing = model._metadata[organism_enum].strand_reindexing
    strand_reindexing = jax.device_put(strand_reindexing, model._device_context._device)

    for batch in val_loader.generate():
        ref = batch['ref_seq']
        alt = batch['alt_seq']
        targets = batch['targets']
        mask = batch['gene_mask']
        organism_index = jnp.zeros((ref.shape[0],), dtype=jnp.int32)
        neg_strand = jnp.zeros((ref.shape[0],), dtype=jnp.bool_)

        pred_ref = model._predict(model._params, model._state, ref, organism_index,
                                   negative_strand_mask=neg_strand,
                                   strand_reindexing=strand_reindexing,
                                   gene_mask=mask)[head_name]
        pred_alt = model._predict(model._params, model._state, alt, organism_index,
                                   negative_strand_mask=neg_strand,
                                   strand_reindexing=strand_reindexing,
                                   gene_mask=mask)[head_name]
        diff = (pred_alt - pred_ref).squeeze(-1)
        loss = jnp.mean((diff - targets) ** 2)
        total_loss += loss
        all_preds.extend(diff)
        all_targets.extend(targets)
        num_batches += 1

    avg_loss = total_loss / num_batches
    if len(all_preds) > 1:
        corr = jnp.corrcoef(jnp.array(all_preds), jnp.array(all_targets))[0, 1]
    else:
        corr = float('nan')
    return avg_loss, corr
