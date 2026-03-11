# create_model.py
import jax
from alphagenome_ft import create_model_with_heads, parameter_utils
from register_head import *

def get_model(model_version='all_folds', head_name='promoter_effect_head', device=None):
    if device is None:
        try:
            device = jax.devices('gpu')[0]
        except:
            try:
                device = jax.devices('tpu')[0]
            except:
                device = jax.devices('cpu')[0]
    model = create_model_with_heads(
        model_version,
        heads=[head_name],
        device=device,
    )
    model._params = parameter_utils.freeze_except_lora(model._params)
    return model
