# Promoter E2E: frozen AlphaGenome + GeneMask LFC + EffectHead

Этот каталог — **PromoterAI**: предсказание **z-score** allelic effect на экспрессию для SNP
в промоторной области. Модель **не** читает заранее сохранённый `train_features.npz`; вместо
этого на **каждом** шаге обучения она сама считает признаки из последовательностей.

Общие модули (`common.py`, `heads.py`, `e2e_lfc.py`, …) лежат в [`../shared/`](../shared/).

---

## Что происходит по шагам (простыми словами)

1. Из CSV берётся строка: хромосома, позиция SNP, ref/alt, gene_id, **таргет z**.
2. Из `hg38.fa` и GENCODE вырезается окно **16 384 п.н.** вокруг варианта и строится **gene mask**
   (какие позиции окна попадают на тело целевого гена).
3. Строятся one-hot **ref** и **alt** (длина окна × 4).
4. **AlphaGenome** (веса **не обновляются**) дважды предсказывает треки **RNA_SEQ** с
   разрешением 1 bp.
5. По маске гена считается **log-fold-change (LFC)** между alt и ref — вектор по RNA-трекам.
6. LFC **стандартизируется** (mean/std, посчитанные один раз на train) и подаётся в **EffectHead**
   (MLP 256→64→1).
7. Loss сравнивает **ẑ** с **z** из CSV; обновляются **только** веса MLP.

Валидация — Pearson r на `val_variants1.csv`. Test — `tableS1A.tsv` (классы over/under/none, без z).

---

## Формулы

> На GitHub формулы пишутся через **`$$ … $$`** (блок) и **`$ … $`** (в строке). Ниже — в этом формате.

### Окно и маска

- Длина окна: **W = 16 384** (параметр `--window`, в launch зафиксировано).
- Центр окна — позиция SNP; цепь учитывается при вырезке из FASTA.

### LFC по GeneMask (как в `extract_features.py`)

Для каждого RNA-трека $t$ и батча вариантов сначала суммируем предсказанную RNA-seq
покрытость по позициям **gene mask** (только неотрицательные значения):

$$
S_{\text{ref}}(t) = \sum_{i \in \text{gene}} \max(R^{\text{ref}}_{i,t},\, 0)
$$

$$
S_{\text{alt}}(t) = \sum_{i \in \text{gene}} \max(R^{\text{alt}}_{i,t},\, 0)
$$

Log-fold-change по треку:

$$
\text{LFC}(t) = \log\bigl(S_{\text{alt}}(t) + \varepsilon\bigr) - \log\bigl(S_{\text{ref}}(t) + \varepsilon\bigr), \quad \varepsilon = 10^{-3}
$$

Далее LFC обрезается и `nan` заменяются (`sanitize_lfc`, clip ±50 при подаче в голову).

**То же словами:** для каждого трека складываем ref/alt сигнал по экзону гена, берём log отношения alt/ref.

### Scaler (один раз на train)

По всем train-строкам (после фильтра gene mask) собирается матрица сырого LFC и считаются
по каждой координате $j$:

$$
\tilde{x}_j = \text{clip}\left( \frac{x_j - \mu_j}{\sigma_j},\, -50,\, 50 \right)
$$

$\mu_j$, $\sigma_j$ сохраняются в `lfc_scaler.npz` в `--out-dir`.

### EffectHead

$$
\hat{z} = W_z\,\phi(W_2\,\phi(W_1\,\tilde{\mathbf{x}} + b_1) + b_2) + b_z
$$

где $\phi$ — ReLU, dropout при train. Promoter E2E **не** использует aux-голову направления
(в batch `p_over` = NaN).

### Loss (Huber)

$$
L_z = \frac{1}{|M|} \sum_{i \in M} \text{Huber}(\hat{z}_i - z_i)
$$

$$
\text{Huber}(r) = \begin{cases} \tfrac{1}{2} r^2 & \text{если } |r| \le \delta \\ \delta\,(|r| - \tfrac{1}{2}\delta) & \text{если } |r| > \delta \end{cases}
$$

$\delta = 1$; $M$ — множество строк с конечным $z$.

---

## Параметры запуска (как в `launch_e2e_head_ft.sh`)

| Параметр | Значение |
|----------|----------|
| `--window` | **16384** |
| `--batch-size` | **8** |
| `--val-batch-size` | **16** |
| `--scaler-batch-size` | **8** |
| `--lr` | **1e-3** |
| `--weight-decay` | **1e-4** (default в скрипте) |
| `--max-epochs` | **200** |
| `--patience` | **15** |
| `--seed` | **42** |
| GPU | **CUDA_VISIBLE_DEVICES=3** |
| JAX VRAM | `PREALLOCATE=true`, `MEM_FRACTION=0.88` |
| Warm-start | `runs/effect_head_v1/best.pkl` (Stage 2) |
| Train CSV | `train_variants1.csv` (~32k строк → после фильтра меньше) |
| Val CSV | `val_variants1.csv` |
| Test | `tableS1A.tsv` |

Ожидаемое время: **часы–сутки** (медленнее Stage 2 на npz). Остановка — early stopping по val Pearson.

---

## Инициализация головы (не читерство)

`--head-checkpoint` загружает **только** `params` MLP, обученные в Stage 2 на **offline** LFC
из того же GeneMask. Test **не** участвует в fit scaler по умолчанию (scaler снова на train E2E).
Это стандартная схема **pretrain → fine-tune** при смене способа получения признаков (npz → online).

---

## Сравнение со старыми подходами

### 1. Курсовой код (`linear_head_*`, `head_mean`, `lora_*` в корне репозитория)

| Старый подход | Проблема | Promoter E2E |
|---------------|----------|--------------|
| `mean(embed)` по 20 kb | SNP-сигнал размывается в \(1/W\) | **LFC по RNA-трекам в теле гена** |
| LoRA всего backbone | нестабильно / OOM | backbone **frozen** |
| MSE на сыром mean | val r **падает** после пика | Huber на **z**, val r **~0.29** (Stage 2) |

### 2. Stage 2 [`best_code`](../best_code/) (ваш рабочий пайплайн на GitHub)

| | Stage 2 `best_code` | Promoter E2E (эта папка) |
|---|----------------------|---------------------------|
| Признак X | `extract_features.py` → npz | тот же **смысл** LFC, но **online** |
| Скорость обучения | минуты–часы | часы–сутки |
| Eval test | npz + `evaluate.py` | `evaluate_e2e_promoter.py` (forward) |
| Метрики-ориентир | val r **0.29**, test AUC **0.81** | цель — **тот же порядок** |

E2E **не** «подкручивает» метрики test: test по-прежнему только для финального eval.

### 3. Random Forest на LFC (`AG/rf.py`)

RF — отдельный классификатор на **тех же** LFC-признаках. EffectHead — **дифференцируемая**
регрессия + ROC на test; Stage 2 уже показал AUC **≥ RF** на promoter test.

---

## Файлы в этой папке

| Файл | Роль |
|------|------|
| `finetune_e2e_head_promoter.py` | Обучение |
| `evaluate_e2e_promoter.py` | Val Pearson + test AUC |
| `promoter_sequence_loader.py` | FASTA, GTF, gene mask |
| `train.py` | `make_batches`, `pearson_r` |

## Запуск только promoter

```bash
cd .. && bash launch_e2e_head_ft.sh promoter
```

Артефакты: `../runs/e2e_head_promoter_v1/best.pkl`, лог `../runs/e2e_head_promoter_v1.log`,
графики `analysis/` (после make_figures).
