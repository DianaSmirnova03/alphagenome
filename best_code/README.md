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

## Графики и выводы (`analysis/`)

<table>
<tr>
<td width="33%" valign="top"><img src="analysis/train_curve.png" alt="train_curve" width="100%"/><br/><strong>train_curve.png</strong><br/><strong>Что:</strong> эпоха → train loss (Huber) и val Pearson r; красная точка — best epoch.<br/><strong>Как:</strong> TensorBoard после <code>train.py</code>; <code>make_figures.py</code>.<br/><strong>Вывод:</strong> loss падает без расходимости; val r ~0.26–0.29 без деградации к концу (в отличие от head на raw embeddings).</td>
<td width="33%" valign="top"><img src="analysis/val_scatter.png" alt="val_scatter" width="100%"/><br/><strong>val_scatter.png</strong><br/><strong>Что:</strong> точки val: истинный <code>z</code> vs pred EffectHead.<br/><strong>Как:</strong> <code>val_features.npz</code>, <code>best.pkl</code>; <code>make_figures.py</code>.<br/><strong>Вывод:</strong> устойчивая слабая связь (r ≈ 0.29); разброс ожидаем — не весь <code>z</code> кодируется seq.</td>
<td width="33%" valign="top"><img src="analysis/val_residual.png" alt="val_residual" width="100%"/><br/><strong>val_residual.png</strong><br/><strong>Что:</strong> остаток pred − <code>z</code> vs <code>z</code> на val.<br/><strong>Как:</strong> те же pred, что scatter; <code>make_figures.py</code>.<br/><strong>Вывод:</strong> нет явной U-формы или смещения на больших |z|; модель не схлопывается в ноль.</td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis/test_roc.png" alt="test_roc" width="100%"/><br/><strong>test_roc.png</strong><br/><strong>Что:</strong> ROC на test (<code>tableS1A</code>): over/none, under/none, over/under; score = ẑ.<br/><strong>Как:</strong> <code>test_features.npz</code>, <code>consequence</code>; как <code>evaluate.py</code> / <code>AG/rf.py</code>.<br/><strong>Вывод:</strong> все AUC &gt; 0.77; over vs under ≈ 0.91 — знак разделяется лучше, чем «эффект vs none».</td>
<td width="33%" valign="top"><img src="analysis/test_pr.png" alt="test_pr" width="100%"/><br/><strong>test_pr.png</strong><br/><strong>Что:</strong> precision–recall для тех же трёх pairwise задач на test.<br/><strong>Как:</strong> <code>make_figures.py</code> (sklearn PR curves).<br/><strong>Вывод:</strong> при дисбалансе PR дополняет ROC; для over/none возможен high-recall при умеренной precision.</td>
<td width="33%" valign="top"><img src="analysis/test_kde.png" alt="test_kde" width="100%"/><br/><strong>test_kde.png</strong><br/><strong>Что:</strong> density ẑ по классам none / over / under на test.<br/><strong>Как:</strong> <code>make_figures.py</code>.<br/><strong>Вывод:</strong> none около 0; over и under смещены в разные стороны — модель несёт знак, не только «есть эффект».</td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis/test_calibration_over.png" alt="test_calibration_over" width="100%"/><br/><strong>test_calibration_over.png</strong><br/><strong>Что:</strong> decile среднего ẑ → доля истинных over в бине.<br/><strong>Как:</strong> сортировка test, 10 бинов; <code>make_figures.py</code>.<br/><strong>Вывод:</strong> монотонный рост «выше ẑ → чаще over» — грубая калибровка знака score.</td>
<td width="33%" valign="top"><img src="analysis/test_auc_rf_vs_head.png" alt="test_auc_rf_vs_head" width="100%"/><br/><strong>test_auc_rf_vs_head.png</strong><br/><strong>Что:</strong> AUC test: RF (min–max по 3 запускам) vs EffectHead.<br/><strong>Как:</strong> checkpoint + диапазоны RF из логов; <code>make_figures.py</code>.<br/><strong>Вывод:</strong> EffectHead не хуже RF на LFC; over/under выше верхней границы RF.</td>
<td width="33%" valign="top"><img src="analysis/metrics_card.png" alt="metrics_card" width="100%"/><br/><strong>metrics_card.png</strong><br/><strong>Что:</strong> текстовая сводка val/test из JSON.<br/><strong>Как:</strong> <code>report_test_metrics.py</code> + <code>make_figures.py</code>.<br/><strong>Вывод:</strong> быстрая сверка цифр README без повторного eval.</td>
</tr>
</table>

### Сравнение со старыми результатами

| Подход | Val Pearson r | AUC over/none | AUC under/none | AUC over/under |
|---|---|---|---|---|
| `linear_head_masked`, `head_mean`, `head_tracks_970` (файнтюнинг, сырые эмбеддинги) | 0.16–0.20 → **деградирует** до 0.09–0.15 | — | — | — |
| `lora_128`, `lora_970` (LoRA на весь backbone) | -0.02…0.08 (шум) / OOM | — | — | — |
| Random Forest на LFC-эмбеддингах (`AG/rf.py`, 3 запуска) | не считался (только классификация) | 0.73–0.78 | 0.75–0.78 | 0.86–0.90 |
| **Новый пайплайн (`EffectHead`, этот репозиторий)** | **0.291** (val) | **0.810** | **0.779** | **0.914** |

```bash
export JAX_PLATFORMS=cpu
python report_test_metrics.py --out-dir results_eval
python make_figures.py --task promoter --promoter-out analysis
```

Вывод по пайплайну: единственный стабильный Pearson r ~0.29 на val без деградации
(у старых head — максимум 0.1–0.2 и последующий распад). При
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
