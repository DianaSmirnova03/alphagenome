# run_training.py
import jax
import optax
from pathlib import Path
from tensorboardX import SummaryWriter

from data_loader import VariantDataset
from create_model import get_model
from train import create_train_step, train_epoch, validate


FASTA_PATH = "../hg38.fa"
EXON_PKL_PATH = "../data/gene_exons.pkl"
TRAIN_CSV = "../data/train_with_genes.csv"
VAL_CSV = "../data/val_with_genes.csv"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 10
HEAD_NAME = 'promoter_effect_head'
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOG_DIR = CHECKPOINT_DIR / 'logs'
writer = SummaryWriter(log_dir=str(LOG_DIR))

device = jax.devices()[0]
train_dataset = VariantDataset(TRAIN_CSV, FASTA_PATH, EXON_PKL_PATH, batch_size=BATCH_SIZE, shuffle=True)
val_dataset   = VariantDataset(VAL_CSV,   FASTA_PATH, EXON_PKL_PATH, batch_size=BATCH_SIZE, shuffle=False)
model = get_model(model_version='all_folds', head_name=HEAD_NAME, device=device)
optimizer = optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
opt_state = optimizer.init(model._params)
train_step = create_train_step(model, optimizer, head_name=HEAD_NAME)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    train_loss, opt_state = train_epoch(model, train_dataset, optimizer, opt_state, train_step)
    val_loss, val_corr = validate(model, val_dataset, HEAD_NAME)
    print(f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, Val corr: {val_corr:.4f}")

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Corr/val', val_corr, epoch)

    model.save_checkpoint(CHECKPOINT_DIR / f'epoch_{epoch}', save_full_model=False)

writer.close()
print("Обучение завершено.")
