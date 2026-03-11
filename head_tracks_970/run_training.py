import jax
import optax
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import csv
import time
from data_loader import PromoterDataset
from create_model import get_model
from train import create_train_step, train_epoch, validate

FASTA_PATH = '../hg38.fa'
TRAIN_CSV = '../data/train_variants.csv'
VAL_CSV = '../data/val_variants.csv'
BATCH_SIZE = 4
BASE_LR = 5e-4               
BASE_WD = 5e-6              
NUM_EPOCHS = 30             
HEAD_NAME = 'promoter_effect_head'
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = CHECKPOINT_DIR / 'logs'
writer = SummaryWriter(log_dir=str(LOG_DIR))
csv_path = CHECKPOINT_DIR / 'logs.csv'
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_corr', 'lr'])

device = jax.devices()[0]
train_dataset = PromoterDataset(TRAIN_CSV, FASTA_PATH, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = PromoterDataset(VAL_CSV, FASTA_PATH, batch_size=BATCH_SIZE, shuffle=False)
model = get_model(model_version='all_folds', head_name=HEAD_NAME, device=device)

def lr_schedule(epoch):
    warmup_epochs = int(0.1 * NUM_EPOCHS)
    decay_epochs = int(0.1 * NUM_EPOCHS)
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    elif epoch > NUM_EPOCHS - decay_epochs:
        return (NUM_EPOCHS - epoch) / decay_epochs
    else:
        return 1.0


schedule_fn = optax.piecewise_constant_schedule(
    init_value=BASE_LR,
    boundaries_and_scales={
        int(0.1 * NUM_EPOCHS): 1.0,  
        int(0.9 * NUM_EPOCHS): 0.1   
    }
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),                   
    optax.adamw(learning_rate=schedule_fn, weight_decay=BASE_WD)
)
opt_state = optimizer.init(model._params)
train_step = create_train_step(model, optimizer, head_name=HEAD_NAME)
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    start_time = time.time()
    train_loss, opt_state = train_epoch(model, train_dataset, optimizer, opt_state, train_step, HEAD_NAME)
    val_loss, val_corr = validate(model, val_dataset, HEAD_NAME)
    current_lr = schedule_fn(epoch) 
    elapsed = time.time() - start_time
    print(f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, Val corr: {val_corr:.4f}, LR: {current_lr:.2e}, Time: {elapsed:.1f}s")

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Corr/val', val_corr, epoch)
    writer.add_scalar('LR', current_lr, epoch)

    def get_head_params_flat(params, head_name):
        head_params = {}
        for path, val in jax.tree_util.tree_flatten_with_path(params)[0]:
            path_str = '/'.join(str(p) for p in path)
            if head_name in path_str:
                head_params[path_str] = val
        return head_params

    head_params = get_head_params_flat(model._params, HEAD_NAME)
    for name, arr in head_params.items():
        writer.add_histogram(f'head/{name}', np.array(arr), epoch)

    csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_corr:.4f}", f"{current_lr:.2e}"])
    csv_file.flush()
    model.save_checkpoint(CHECKPOINT_DIR / f'epoch_{epoch}', save_full_model=False)
writer.close()
csv_file.close()
