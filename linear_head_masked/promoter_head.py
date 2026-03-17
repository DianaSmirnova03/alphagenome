import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead

class PromoterEffectHead(CustomHead):
    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(name=name, num_tracks=num_tracks,
                         output_type=output_type,
                         num_organisms=num_organisms, metadata=metadata)
        self.dropout_rate = metadata.get('dropout_rate', 0.0)

    def predict(self, embeddings, organism_index, gene_mask=None, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
      #  if gene_mask is not None:
      #      masked = x * gene_mask[..., None]
      #      sum_masked = jnp.sum(masked, axis=1)
      #      mask_sum = jnp.sum(gene_mask, axis=1, keepdims=True) + 1e-8
      #      x_agg = sum_masked / mask_sum
      #  else:
      #      x_agg = jnp.mean(x, axis=1)
        x_agg = jnp.mean(x, axis=1)
        out = hk.Linear(1, name='output')(x_agg)
        return out

    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        mse = jnp.mean((predictions.squeeze(-1) - targets) ** 2)
        return {'loss': mse, 'mse': mse}
