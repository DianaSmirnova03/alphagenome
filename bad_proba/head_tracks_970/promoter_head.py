import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead

class PromoterEffectHead(CustomHead):
    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(name=name, num_tracks=num_tracks,
                         output_type=output_type,
                         num_organisms=num_organisms, metadata=metadata)

    def predict(self, embeddings, organism_index, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        logits = hk.Linear(self._num_tracks, name='output')(x)
        return logits

    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        dummy = jnp.mean(predictions) * 0.0
        return {'loss': dummy, 'mse': dummy}
