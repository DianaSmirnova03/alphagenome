import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead, lora

class PromoterEffectHead(CustomHead):
    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(name=name, num_tracks=num_tracks,
                         output_type=output_type,
                         num_organisms=num_organisms, metadata=metadata)
        self.lora_rank = metadata.get('lora_rank', 8)
        self.lora_alpha = metadata.get('lora_alpha', self.lora_rank)
        self.dropout_rate = metadata.get('dropout_rate', 0.0)

    def predict(self, embeddings, organism_index, **kwargs):
        x = embeddings.get_sequence_embeddings(resolution=1)
        x = jnp.mean(x, axis=1)
        if self.dropout_rate > 0:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)

        #линейная проекция с LoRA на num_tracks каналов
        cfg = lora.LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)
        out = lora.LoRALinear(self._num_tracks, cfg, name='output')(x)

        return out

    def loss(self, predictions, batch):
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        dummy = jnp.mean(predictions) * 0.0
        return {'loss': dummy, 'mse': dummy}
