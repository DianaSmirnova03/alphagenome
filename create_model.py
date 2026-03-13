import jax
from alphagenome_ft import create_model_with_heads
from register_head import *

def get_model(model_version='all_folds', head_name='promoter_effect_head', device=None):
    if device is None:
        device = jax.devices()[0]
    model = create_model_with_heads(
        model_version,
        heads=[head_name],
        device=device,
    )
    model.freeze_except_head(head_name)
    return model
