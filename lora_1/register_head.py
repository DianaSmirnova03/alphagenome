# register_head.py
from alphagenome.models import dna_output
from alphagenome_ft import (
    CustomHeadConfig,
    CustomHeadType,
    register_custom_head
)
from promoter_head import PromoterEffectHead

register_custom_head(
    'promoter_effect_head',
    PromoterEffectHead,
    CustomHeadConfig(
        type=CustomHeadType.GENOME_TRACKS,
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
        metadata={
            'lora_rank': 4,                   
            'lora_alpha': 8,                   
            'dropout_rate': 0.3                   
        }
    )
)
print("LoRA-голова 'promoter_effect_head' зарегистрирована.")
