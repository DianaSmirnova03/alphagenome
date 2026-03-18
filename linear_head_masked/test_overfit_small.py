import jax.numpy as jnp
import jax
import optax
import numpy as np
import matplotlib.pyplot as plt
from data_loader import VariantDataset
from create_model import get_model
from train import create_train_step, train_epoch, validate

HEAD_NAME = 'promoter_effect_head'

FASTA_PATH = "../hg38.fa"
EXON_PKL_PATH = "../data/gene_exons.pkl"
TRAIN_CSV = "../data/train_with_genes.csv"
BATCH_SIZE = 2
SMALL_SIZE = 10                
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 200
PATIENCE = 5
PLOT_FILE = "overfit_loss.png"

print("=" * 60)
print("ТЕСТ ПЕРЕОБУЧЕНИЯ")
print("=" * 60)

# Датасет
ds = VariantDataset(TRAIN_CSV, FASTA_PATH, EXON_PKL_PATH, batch_size=BATCH_SIZE, shuffle=True)
ds.items = ds.items[:SMALL_SIZE]
print(f"Обучающих примеров: {len(ds.items)}")
print(f"Таргеты (первые 5): {[item['z'] for item in ds.items[:5]]}")
print(f"Среднее таргетов: {np.mean([item['z'] for item in ds.items]):.4f}, стд: {np.std([item['z'] for item in ds.items]):.4f}")

val_ds = VariantDataset(TRAIN_CSV, FASTA_PATH, EXON_PKL_PATH, batch_size=BATCH_SIZE, shuffle=False)
val_ds.items = val_ds.items[:SMALL_SIZE]
 
device = jax.devices()[0]
model = get_model(model_version='all_folds', head_name=HEAD_NAME, device=device)
 
params_before = jax.tree_util.tree_map(lambda x: x.copy(), model._params)
 
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
)
opt_state = optimizer.init(model._params)
train_step = create_train_step(model, optimizer, head_name=HEAD_NAME)
 
def predict_all(model, dataset):
    all_preds = []
    all_targets = []
    for batch in dataset.generate():
        pred_ref = model._predict(
            model._params, model._state,
            jax.device_put(batch['ref_seq']),
            jnp.zeros((batch['ref_seq'].shape[0],), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros((batch['ref_seq'].shape[0],), dtype=jnp.bool_),
            strand_reindexing=jax.device_put(model._metadata[next(iter(model._metadata.keys()))].strand_reindexing, model._device_context._device),
            gene_mask=jax.device_put(batch['gene_mask'])
        )[HEAD_NAME]
        pred_alt = model._predict(
            model._params, model._state,
            jax.device_put(batch['alt_seq']),
            jnp.zeros((batch['alt_seq'].shape[0],), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros((batch['alt_seq'].shape[0],), dtype=jnp.bool_),
            strand_reindexing=jax.device_put(model._metadata[next(iter(model._metadata.keys()))].strand_reindexing, model._device_context._device),
            gene_mask=jax.device_put(batch['gene_mask'])
        )[HEAD_NAME]
        diff = (pred_alt - pred_ref).squeeze(-1)
        all_preds.extend(np.array(diff))
        all_targets.extend(batch['targets'])
    return np.array(all_preds), np.array(all_targets)

print("Предсказания до обучения:")
preds_before, targets_before = predict_all(model, ds)
print(f"  Предсказанные разности: {preds_before}")
print(f"  Целевые значения: {targets_before}")
print(f"  MSE до обучения: {np.mean((preds_before - targets_before)**2):.6f}")

print("Начинаем обучение...")
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, opt_state = train_epoch(model, ds, optimizer, opt_state, train_step)
    val_loss, val_corr = validate(model, val_ds, HEAD_NAME)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
 
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_corr={val_corr:.4f}")
 
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Ранняя остановка на эпохе {epoch} (patience {PATIENCE})")
            break
 
preds_after, targets_after = predict_all(model, ds)
print("\nПредсказания после обучения:")
print(f"  Предсказанные разности: {preds_after}")
print(f"  Целевые значения: {targets_after}")
print(f"  MSE после обучения: {np.mean((preds_after - targets_after)**2):.6f}")
 
params_after = model._params

def get_head_params_diff(p1, p2, head_name):
    diff_norm = 0.0
    for (path1, v1), (path2, v2) in zip(
        jax.tree_util.tree_flatten_with_path(p1)[0],
        jax.tree_util.tree_flatten_with_path(p2)[0]
    ):
        path_str = '/'.join(str(p) for p in path1)
        if head_name in path_str:
            diff_norm += np.linalg.norm(np.array(v2) - np.array(v1))
    return diff_norm

head_diff = get_head_params_diff(params_before, params_after, HEAD_NAME)
print(f"\nИзменение параметров головы (суммарная L2 норма разности): {head_diff:.6f}")
if head_diff > 1e-6:
    print("✓ Параметры головы изменились – обучение идёт.")
else:
    print("⚠️ Параметры головы почти не изменились – возможно, градиенты малы.")
 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Переобучение на малом наборе данных')
plt.legend()
plt.grid(True)
plt.savefig(PLOT_FILE)
print(f"График сохранён в {PLOT_FILE}")

print("\nТест завершён.")
