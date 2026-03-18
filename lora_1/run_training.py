import jax
import optax
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
from data_loader import PromoterDataset
from create_model import get_model
from train import create_train_step, train_epoch, validate

FASTA_PATH = '../hg38.fa'
TRAIN_CSV = '../data/train_variants.csv'
VAL_CSV = '../data/val_variants.csv'
BATCH_SIZE = 4
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-5                        
NUM_EPOCHS = 20                           
HEAD_NAME = 'promoter_effect_head'
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOG_DIR = CHECKPOINT_DIR / 'logs'
writer = SummaryWriter(log_dir=str(LOG_DIR))

device = jax.devices()[0]
train_dataset = PromoterDataset(TRAIN_CSV, FASTA_PATH, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = PromoterDataset(VAL_CSV, FASTA_PATH, batch_size=BATCH_SIZE, shuffle=False)

model = get_model(model_version='all_folds', head_name=HEAD_NAME, device=device)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
)
opt_state = optimizer.init(model._params)
train_step = create_train_step(model, optimizer, head_name=HEAD_NAME)

#проверка
def get_trainable_params(params):
    trainable = []
    def check(path, value):
        path_str = '/'.join(str(p) for p in path)
        if 'lora' in path_str:
            trainable.append(path_str)
    jax.tree_util.tree_map_with_path(check, params)
    return trainable
for p in get_trainable_params(model._params):
    print(" ", p)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    train_loss, opt_state = train_epoch(model, train_dataset, optimizer, opt_state, train_step, HEAD_NAME)
    val_loss, val_corr = validate(model, val_dataset, HEAD_NAME)
    print(f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, Val corr: {val_corr:.4f}")

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Corr/val', val_corr, epoch)
    writer.add_scalar('LR', LEARNING_RATE, epoch)
    model.save_checkpoint(CHECKPOINT_DIR / f'epoch_{epoch}', save_full_model=False)

writer.close()
