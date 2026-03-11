import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead
from alphagenome_research.model import layers 

class PromoterEffectHead(CustomHead):
    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(name=name, num_tracks=num_tracks,
                         output_type=output_type,
                         num_organisms=num_organisms, metadata=metadata)
        self.hidden_size = metadata.get('hidden_size', 256) if metadata else 256

    def predict(self, embeddings, organism_index, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        x = jnp.mean(x, axis=1)
        x = hk.Linear(self.hidden_size, name='hidden')(x)
        x = jax.nn.relu(x)
        out = hk.Linear(1, name='output')(x)
        return out

    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        mse = jnp.mean((predictions - targets) ** 2)
        return {'loss': mse, 'mse': mse}
