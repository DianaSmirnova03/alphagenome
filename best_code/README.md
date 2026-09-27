# AlphaGenome SNP-эффект: файнтюнинг на собственных данных

Этот пайплайн — переписанный с нуля, рабочий вариант дообучения AlphaGenome на
ваших данных (референс/альтернативные последовательности SNP + z-score
эффекта на экспрессию, `train_variants*.csv` / `val_variants*.csv` /
`tableS1A.tsv`).

## Результаты

Сводка метрик: [`results_test_metrics.json`](results_test_metrics.json).  
Чекпоинт для всех оценок: `runs/effect_head_v1/best.pkl`.  
Признаки: `features/{train,val,test}_features.npz` (GeneMask LFC, 667 dim).  
Пересчёт PNG: `make_figures.py --task promoter --promoter-out analysis` (нужен TensorBoard-лог в `runs/effect_head_v1/tensorboard`).

### Таблица метрик

| Split | Метрика | Значение |
|-------|---------|----------|
| val (n=2044) | Pearson r (`z`) | **0.291** |
| test (n=749) | AUC over / none | **0.810** |
| test | AUC under / none | **0.779** |
| test | AUC over / under | **0.914** |

На test нет числового `z` — только класс `consequence` (over / under / none).

---

## Описание графиков (`analysis/`)

#### `train_curve.png`

**Содержание:** по оси X — номер эпохи; синяя кривая — средний train loss (Huber по `z`); оранжевая — Pearson r на val; красная точка — эпоха с максимальным val r (по ней сохранён `best.pkl`).

**Как получен:** скаляры `train/loss` и `val/pearson_r` из TensorBoard после `train.py`; рисует `make_figures.py`.

**Вывод:** loss снижается без «взрыва»; val r выходит на плато ~0.26–0.29 и не обнуляется к концу обучения (в отличие от старых head на raw embeddings).

---

#### `val_scatter.png`

**Содержание:** каждая точка — вариант из `val_features.npz`; оси: истинный `z` и предсказание EffectHead.

**Как получен:** forward `best.pkl` на масштабированном LFC; `make_figures.py`.

**Вывод:** есть слабая, но устойчивая линейная связь (r ≈ 0.29); большой разброс ожидаем — часть эффекта в `z` не кодируется одной последовательностью.

---

#### `val_residual.png`

**Содержание:** остаток `pred − z` против истинного `z` на val.

**Как получен:** те же предсказания, что для scatter; `make_figures.py`.

**Вывод:** остатки не систематически U-образные; нет явного смещения только на больших |z|, модель не «залипает» в ноль.

---

#### `test_roc.png`

**Содержание:** три ROC-кривые на held-out `tableS1A` (test): over vs none, under vs none, over vs under; score — предсказанный ẑ.

**Как получен:** `test_features.npz`, метки `consequence`; `make_figures.py` (та же постановка, что `AG/rf.py` / `evaluate.py`).

**Вывод:** все три AUC > 0.77; разделение направления over/under (0.91) сильнее, чем «эффект vs none».

---

#### `test_pr.png`

**Содержание:** precision–recall для тех же трёх pairwise задач на test.

**Как получен:** `make_figures.py`, sklearn `precision_recall_curve`.

**Вывод:** при дисбалансе классов PR дополняет ROC; high-recall режим для over vs none достижим при умеренной precision (см. кривые на рисунке).

---

#### `test_kde.png`

**Содержание:** нормированные гистограммы (density) предсказанного ẑ отдельно для классов none / over / under на test.

**Как получен:** `make_figures.py`.

**Вывод:** none сосредоточен около 0; over и under смещены в разные стороны — модель несёт информацию о знаке, не только о «наличии эффекта».

---

#### `test_calibration_over.png`

**Содержание:** по decile среднего ẑ на test (ось X) — доля истинных `over` в этом decile (ось Y).

**Как получен:** сортировка test по ẑ, 10 бинов; `make_figures.py`.

**Вывод:** монотонный тренд «выше ẑ → чаще over» говорит о калиброванности знака score для класса over (грубая, без Platt scaling).

---

#### `test_auc_rf_vs_head.png`

**Содержание:** столбцы AUC на test: RF (полоска ошибок — min/max по трём запускам из `AG/rf.py`) vs EffectHead (один столбец).

**Как получен:** EffectHead — из текущего checkpoint; RF — фиксированные диапазоны из README/логов RF; `make_figures.py`.

**Вывод:** EffectHead на том же типе LFC-признаков не хуже RF по всем трём парам; на over/under даже выше верхней границы RF.

---

#### `metrics_card.png`

**Содержание:** текстовая сводка ключевых val/test метрик из `results_test_metrics.json`.

**Как получен:** `make_figures.py` после `report_test_metrics.py`.

**Вывод:** быстрая проверка цифр в README без запуска eval.

---

### Сравнение со старыми результатами

| Подход | Val Pearson r | AUC over/none | AUC under/none | AUC over/under |
|---|---|---|---|---|
| `linear_head_masked`, `head_mean`, `head_tracks_970` (файнтюнинг, сырые эмбеддинги) | 0.16–0.20 → **деградирует** до 0.09–0.15 | — | — | — |
| `lora_128`, `lora_970` (LoRA на весь backbone) | -0.02…0.08 (шум) / OOM | — | — | — |
| Random Forest на LFC-эмбеддингах (`AG/rf.py`, 3 запуска) | не считался (только классификация) | 0.73–0.78 | 0.75–0.78 | 0.86–0.90 |
| **Новый пайплайн (`EffectHead`, этот репозиторий)** | **0.291** (val) | **0.810** | **0.779** | **0.914** |

### Галерея (клик — файл)

| | | |
|:---:|:---:|:---:|
| [train](analysis/train_curve.png) | [val scatter](analysis/val_scatter.png) | [val residual](analysis/val_residual.png) |
| [test ROC](analysis/test_roc.png) | [test PR](analysis/test_pr.png) | [test KDE](analysis/test_kde.png) |
| [calibration](analysis/test_calibration_over.png) | [RF vs head](analysis/test_auc_rf_vs_head.png) | [metrics](analysis/metrics_card.png) |

Полная сводка val/test (JSON): [`results_test_metrics.json`](results_test_metrics.json).

```bash
export JAX_PLATFORMS=cpu
python report_test_metrics.py --out-dir results_eval
python make_figures.py --task promoter --promoter-out analysis
```

Вывод по пайплайну:
**неслучайную, не деградирующую регрессию** (`z`, Pearson r=0.29 против
максимум 0.1–0.2 и последующего распада у всех вариантов файнтюнинга). При
этом на классификации (over/under/none) — задаче, где раньше единственным
рабочим подходом был Random Forest на тех же LFC-признаках (AUC 0.73–0.90) —
новый пайплайн **сравнялся или превзошёл** RF по всем трём парам сравнения,
уже будучи при этом сквозной, дифференцируемой моделью, которую можно
дообучать дальше (Stage 3), в отличие от отдельно стоящего RF.

## Почему старый код (`my_data/my/*`, `AG/linear_head_*`, `AG/lora_*`, `AG/head_tracks_*`) не обучался

Кратко (подробный разбор — в чате):

1. **Все старые головы усредняли raw-эмбеддинги AlphaGenome по всему окну
   20 480 п.н.** (`jnp.mean(x, axis=1)` или masked-mean по экзонам). Однонуклеотидная
   замена влияет на представление только локально, поэтому разница
   `mean(embed_alt) - mean(embed_ref)` после такого усреднения оказывается в
   тысячи раз меньше сигнала в точке мутации — свежей линейной голове почти
   нечего учить. Это подтверждается логами: train loss монотонно падает
   (голова оптимизируется), а val-корреляция не растёт, а падает
   (`AG/linear_head_masked/run_training.log`, `AG/head_mean/run_training.log` и т.д.)
   — классический оверфит на шум, а не обучение сигнала.
2. **`head_tracks_128/970`** — голова выдаёт 128/970 треков, а лосс усредняет
   их в один скаляр (`jnp.mean(diff, axis=-1)`), хотя эти треки — разные,
   не связанные друг с другом биологические сигналы. Отсюда точное
   переобучение без обобщения (`AG/head_tracks_970/run_training.log`).
3. **LoRA по всему backbone** (`parameter_utils.freeze_except_lora`) при LR
   5e-4–1e-3 без разогрева — расходится (`AG/bad_proba/lora_128/run_training.log`:
   loss улетает в тысячи) или OOM (`AG/lora_1/run_training.log`).
4. Мёртвый код `loss()` вида `dummy = jnp.mean(predictions) * 0.0` в нескольких
   LoRA-головах — всегда нулевой градиент, если этот путь когда-нибудь
   используется.
5. Нет проверки, что `fasta[chrom][pos]` реально совпадает с `ref` из CSV —
   риск сдвига координат (0- vs 1-based) в ручном коде `_get_seq`.
6. То, что **сработало** (`AG/rf.py`, `AG/emb_jax_2304.py`, AUC 0.78–0.90) —
   использовало не сырые эмбеддинги, а **уже откалиброванные предсказания
   выходных RNA_SEQ треков** модели (то, что она реально умеет предсказывать),
   агрегированные как log fold change alt/ref по маске гена — то есть
   фактически официальный `GeneMaskLFCScorer` из `alphagenome_research`.

## Архитектура нового пайплайна

Максимально переиспользует официальный код из `my_data/alphagenome_research`
(тот же пакет, что установлен как `alphagenome_research` в conda-env `ag`) и
`my_data/alphagenome0` (пакет `alphagenome`):

```
Reference seq ──┐                                             ┌─ log(alt/ref) по каждому
                 ├─► AlphaGenome (frozen backbone + RNA_SEQ head) ─┤  RNA_SEQ треку, замаскированному
Alternate seq ──┘        apply_fn(params, state, seq, organism)   │  экзонами целевого гена
                                                                    │  (официальная формула
                                                                    │  GENE_MASK_LFC)
                                                                    ▼
                                                     ┌─────────────────────────┐
                                                     │ Обучаемая голова        │
                                                     │ EffectHead (MLP, JAX)   │──► ẑ (регрессия)
                                                     │ ~10³–10⁴ параметров     │
                                                     └─────────────────────────┘
```

Ключевая идея: модель **явно видит разницу между ref и alt** — это ровно
`log(alt) - log(ref)` на уровне уже обученных, откалиброванных
RNA-seq-предсказаний (а не на уровне усреднённых по всему окну сырых
эмбеддингов). Голова учится ставить в соответствие этому вектору разниц по
трекам ваш таргет `z`. Это именно то представление, которое дало AUC 0.78–0.9
у вас в `rf.py`, только теперь голова — не Random Forest, а обучаемая (через
backprop) сеть, и есть возможность пойти дальше и дообучить последний слой
самой модели (Stage 3, опционально).

### Этапы

1. **`prepare_gtf.py`** — конвертирует GENCODE GTF → `.feather` в формате,
   который понимает `alphagenome_research.model.variant_scoring.gene_mask_extractor.GeneMaskExtractor`
   (нужен один раз).
2. **`extract_features.py`** — для каждого варианта в CSV вызывает
   **официальный** `model.score_variant(interval, variant,
   variant_scorers=[GeneMaskLFCScorer(RNA_SEQ)])` (тот же код, что и в вашем
   рабочем `rf.py`/`emb_jax_2304.py`) и кэширует вектор log-fold-change по
   RNA_SEQ трекам, замаскированный по экзонам нужного гена → `.npz`.
3. **`heads.py`** — маленькая обучаемая голова (JAX/Optax) поверх этого
   вектора: `EffectHead` (2-слойный MLP с dropout) → предсказание `z`.
4. **`train.py`** — обучение головы (Huber-лосс на `z`, опциональная
   вспомогательная BCE-голова на `p_over`), early stopping по Pearson r на
   валидации, чекпоинты.
5. **`evaluate.py`** — метрики на валидации (Pearson/Spearman r, MSE) и на
   `tableS1A.tsv` (AUC over/none, under/none, over/under — те же метрики, что
   в `rf.py`, для честного сравнения).
6. **`finetune_backbone.py`** (опционально, продвинутый режим) — настоящее
   дообучение весов: размораживает последний линейный слой RNA_SEQ-головы
   AlphaGenome (несколько тысяч параметров, не весь backbone) и обучает его
   совместно с `EffectHead` через `jax.grad` напрямую через
   `apply_fn` (дифференцируемый forward, реюзается из
   `alphagenome_research.model.dna_model.create_model`). Используйте только
   после того, как Stage 1+2 отработает и даст baseline — риск переобучения
   выше, LR должен быть маленьким (см. README внутри скрипта).

## Установка

```bash
conda activate ag   # у вас уже есть готовое окружение с alphagenome_research, alphagenome_ft и т.п.
pip install -r requirements.txt
```

## Данные, которые нужны

- `hg38.fa` (+ `.fai`) — у вас уже есть, путь передаётся как `--fasta`.
- `gencode.v39.annotation.gtf` — уже используется в `my_data/my/linear_head_masked/prepare_data.py`.
- `train_variants1.csv`, `val_variants1.csv` (или `train_variants.csv`/`val_variants.csv`)
  — колонки `chrom,pos,ref,alt,gene,gene_id,strand,z,...` (как в `AG/data/`).
- `tableS1A.tsv` — held-out тест с колонками `chrom,pos,ref,alt,strand,gene,consequence`.

## Запуск (полный пайплайн)

```bash
cd DIPLOM/alphagenome_snp_finetune

# 1. Один раз: подготовить GTF в формате feather
python prepare_gtf.py \
    --gtf /mnt/calc/homes/d.smirnova/DIPLOM/gencode.v39.annotation.gtf \
    --out gencode.v39.feather

# 2. Извлечь LFC-признаки (официальный GeneMaskLFCScorer) для train/val/test
python extract_features.py \
    --csv /mnt/calc/homes/d.smirnova/DIPLOM/AG/data/train_variants1.csv --split train \
    --fasta /mnt/calc/homes/d.smirnova/hg38.fa --gtf-feather gencode.v39.feather \
    --out-dir features

python extract_features.py \
    --csv /mnt/calc/homes/d.smirnova/DIPLOM/AG/data/val_variants1.csv --split val \
    --fasta /mnt/calc/homes/d.smirnova/hg38.fa --gtf-feather gencode.v39.feather \
    --out-dir features

python extract_features.py \
    --csv /mnt/calc/homes/d.smirnova/DIPLOM/AG/data/tableS1A.tsv --split test --csv-sep '\t' \
    --fasta /mnt/calc/homes/d.smirnova/hg38.fa --gtf-feather gencode.v39.feather \
    --out-dir features

# 3. Обучить голову
python train.py --features-dir features --out-dir runs/effect_head_v1

# 4. Оценить (val: Pearson/Spearman r, MSE; test: AUC over/none, under/none, over/under)
python evaluate.py --features-dir features --checkpoint runs/effect_head_v1/best.pkl
```

## Опциональный продвинутый Stage 3

```bash
python inspect_params.py --pattern rna_seq   # найти точное имя последнего слоя RNA_SEQ-головы
python finetune_backbone.py --features-dir features --unfreeze-pattern "<имя из шага выше>" \
    --lr 1e-5 --out-dir runs/backbone_ft_v1
```

## Частые проблемы

- Если `score_variant` не находит ген (`GENE_MASK_LFC` вернул пустой AnnData)
  — увеличьте `--window` (по умолчанию 131072 п.н., как в вашем рабочем
  `emb_jax_2304.py`). Для этого датасета `tss_dist` лежит в [-500, 500], так
  что окно 131072 п.н. гарантированно покрывает и вариант, и TSS.
- Kaggle-креды нужны так же, как и в ваших старых скриптах
  (`kagglehub.login()` при первом запуске).
