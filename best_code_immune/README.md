# AlphaGenome на иммунных ASE-данных AIDA

Пайплайн для предсказания **allele-specific expression (ASE)** по данным
Asian Immune Diversity Atlas (AIDA): однонуклеотидные варианты в
периферических мононуклеарных клетках, таргет **`comb_es`** (combined effect
size из MIXALIME) и **`fdr_comb_pval`**, **37 типов иммунных клеток**
(`cell_type`).

Источник данных: `chr_data/chr_train.h5`, `chr_data/chr_test.h5`.

---

## Результаты (рекомендуемая модель: `train_immune_v2.py`)

### Метрики на held-out тесте (`chr_test.h5`)

| Метрика | v1 (`train_immune.py`) | **v2 (`train_immune_v2.py`)** | Курсовая: RF на эмбеддингах |
|---|---|---|---|
| Pearson r, все строки | 0.185 | **0.187** | — |
| Pearson r, только FDR &lt; 0.05 | 0.276 | **0.331** | — |
| AUC direction (over/under), все | 0.601 | **0.608** | — |
| **AUC direction, FDR &lt; 0.05** | 0.694 | **0.766** | **0.763** |

**Direction** — бинарная задача из курсовой: `comb_es > 0` → under,
`comb_es < 0` → over (знак эффекта аллеля).

Валидация (отложенная хромосома **chr14**): Pearson r = **0.329** (v2);
на подмножестве FDR &lt; 0.05 на вале — r ≈ **0.62**.

> **Важно про AUC «значимости ASE» (~0.52 у v1):** предсказывать
> `fdr_comb_pval < 0.05` только по последовательности **нельзя** — FDR
> зависит от глубины покрытия и числа клеток/доноров, а не от ДНК-контекста.
> В v2 aux-голова переобучена на **direction**, как в курсовой.

JSON: [`results_test_metrics.json`](results_test_metrics.json).  
Чекпоинт v2: `checkpoints/effect_head_immune_v2/best.pkl` (v1 — `checkpoints/effect_head_immune_v1/best.pkl`).  
Признаки: `features_immune/*_features.npz` (CenterMask LFC + `cell_type_idx` в Stage 2).

```bash
export JAX_PLATFORMS=cpu
python report_test_metrics.py --immune-features-dir features_immune --out-dir results_eval
python make_figures.py --task immune --immune-out analysis_v2
python evaluate_immune.py --features-dir features_immune \
  --checkpoint checkpoints/effect_head_immune_v1/best.pkl --plots-dir analysis
```

---

## Графики v2 и выводы (`analysis_v2/`)

<table>
<tr>
<td width="33%" valign="top"><img src="analysis_v2/train_curve.png" alt="train_curve" width="100%"/><br/><strong>train_curve.png</strong><br/><strong>Что:</strong> train loss (Huber + FDR-веса) и val Pearson r; красная точка — best epoch.<br/><strong>Как:</strong> TensorBoard v2 → <code>make_figures.py</code>.<br/><strong>Вывод:</strong> val r ~0.14→~0.33 к best; early stop epoch 47 без обвала после плато.</td>
<td width="33%" valign="top"><img src="analysis_v2/val_scatter_immune_v2.png" alt="val_scatter" width="100%"/><br/><strong>val_scatter_immune_v2.png</strong><br/><strong>Что:</strong> <code>comb_es</code> vs pred на chr14 (4497 строк).<br/><strong>Как:</strong> val-маска chr14, checkpoint v2.<br/><strong>Вывод:</strong> r ≈ 0.33; большой разброс из-за 37 cell types и шумных FDR.</td>
<td width="33%" valign="top"><img src="analysis_v2/test_roc_direction_v2.png" alt="test_roc" width="100%"/><br/><strong>test_roc_direction_v2.png</strong><br/><strong>Что:</strong> ROC direction (знак <code>comb_es</code>) на test: all vs FDR&lt;0.05.<br/><strong>Как:</strong> <code>test_features.npz</code>; <code>make_figures.py</code>.<br/><strong>Вывод:</strong> AUC ~0.61 (all) vs **~0.77** (FDR&lt;0.05) — на par с RF курсовой (0.763).</td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis_v2/test_pr_direction_fdr005.png" alt="test_pr" width="100%"/><br/><strong>test_pr_direction_fdr005.png</strong><br/><strong>Что:</strong> PR для direction только FDR&lt;0.05 (n≈1072).<br/><strong>Как:</strong> <code>make_figures.py</code>.<br/><strong>Вывод:</strong> компромисс precision/recall; AP согласуется с AUC direction на subset.</td>
<td width="33%" valign="top"><img src="analysis_v2/test_scatter_all.png" alt="test_scatter_all" width="100%"/><br/><strong>test_scatter_all.png</strong><br/><strong>Что:</strong> scatter <code>comb_es</code> vs pred, все 18766 строк test.<br/><strong>Как:</strong> <code>make_figures.py</code>.<br/><strong>Вывод:</strong> r ≈ 0.19 — много строк с ненадёжной ASE-меткой.</td>
<td width="33%" valign="top"><img src="analysis_v2/test_scatter_fdr005.png" alt="test_scatter_fdr" width="100%"/><br/><strong>test_scatter_fdr005.png</strong><br/><strong>Что:</strong> тот же scatter только FDR&lt;0.05.<br/><strong>Как:</strong> <code>make_figures.py</code>.<br/><strong>Вывод:</strong> r ≈ **0.33** — модель полезнее там, где метка достовернее.</td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis_v2/test_pearson_by_cell_type_v2.png" alt="pearson_by_ct" width="100%"/><br/><strong>test_pearson_by_cell_type_v2.png</strong><br/><strong>Что:</strong> bar Pearson r по <code>cell_type</code> (n≥10).<br/><strong>Как:</strong> группировка test; как <code>evaluate_immune_v2.py</code>.<br/><strong>Вывод:</strong> B-клетки выше; редкие T-подтипы нестабильны из-за малого n.</td>
<td width="33%" valign="top"><img src="analysis_v2/test_heatmap_cell_type.png" alt="heatmap" width="100%"/><br/><strong>test_heatmap_cell_type.png</strong><br/><strong>Что:</strong> heatmap: cell type × (r, direction AUC).<br/><strong>Как:</strong> та же таблица; <code>make_figures.py</code>.<br/><strong>Вывод:</strong> видно, где ловится знак (AUC) vs слабая регрессия величины (r).</td>
<td width="33%" valign="top"><img src="analysis_v2/test_hist_abs_fdr005.png" alt="hist_abs" width="100%"/><br/><strong>test_hist_abs_fdr005.png</strong><br/><strong>Что:</strong> density |<code>comb_es</code>| vs |pred| при FDR&lt;0.05.<br/><strong>Как:</strong> <code>make_figures.py</code>.<br/><strong>Вывод:</strong> pred сжимает хвосты (regression to mean), порядок величин согласован.</td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis_v2/metrics_card.png" alt="metrics_card" width="100%"/><br/><strong>metrics_card.png</strong><br/><strong>Что:</strong> сводка val/test из JSON.<br/><strong>Как:</strong> <code>report_test_metrics.py</code> + <code>make_figures.py</code>.<br/><strong>Вывод:</strong> контроль цифр README одной картинкой.</td>
</tr>
</table>

## Графики v1 и выводы (`analysis/`)

<table>
<tr>
<td width="33%" valign="top"><img src="analysis/val_scatter_immune.png" alt="val_scatter_v1" width="100%"/><br/><strong>val_scatter_immune.png</strong><br/><strong>Что:</strong> chr14 val, pred vs <code>comb_es</code> для v1 (aux на <code>is_sig</code>).<br/><strong>Как:</strong> <code>evaluate_immune.py</code>, v1 checkpoint.<br/><strong>Вывод:</strong> val r all чуть выше v2 (~0.34), но aux некорректен для сравнения с курсовой.</td>
<td width="33%" valign="top"><img src="analysis/test_roc_immune.png" alt="test_roc_v1" width="100%"/><br/><strong>test_roc_immune.png</strong><br/><strong>Что:</strong> ROC «значимость ASE» (|pred| vs FDR&lt;0.05), не direction.<br/><strong>Как:</strong> <code>evaluate_immune.py</code>.<br/><strong>Вывод:</strong> AUC ~0.52 — почти случайно; **не** главная метрика (FDR не из seq).</td>
<td width="33%" valign="top"><img src="analysis/test_pearson_by_cell_type.png" alt="pearson_by_ct_v1" width="100%"/><br/><strong>test_pearson_by_cell_type.png</strong><br/><strong>Что:</strong> Pearson r по cell types на test (v1).<br/><strong>Как:</strong> <code>evaluate_immune.py</code>.<br/><strong>Вывод:</strong> та же heterogeneity, что у v2; r напрямую не сопоставимы без одного loss.</td>
</tr>
</table>

---

## Как работает код (схема)

```mermaid
flowchart TB
  subgraph data [Данные AIDA HDF5]
    H5["chr_train.h5 / chr_test.h5\nchrom, pos, ref, alt\ncell_type, comb_es, fdr_comb_pval"]
  end

  subgraph stage1 [Stage 1 — экстракция признаков GPU]
    FASTA["hg38.fa + genome.Variant\npos 0-based BED → +1"]
    DEDUP["Дедупликация по\nchrom,pos,ref,alt"]
    AG["AlphaGenome frozen\nscore_variant"]
    CMS["CenterMaskScorer\nRNA_SEQ, width=2001\nDIFF_LOG2_SUM"]
    X["X: вектор LFC по трекам\n~667 dim на вариант"]
    H5 --> FASTA --> DEDUP --> AG --> CMS --> X
  end

  subgraph stage2 [Stage 2 — обучение головы CPU/GPU]
    CT["one-hot cell_type\n37+1 dim"]
    HEAD["EffectHead MLP\nHuber(comb_es)"]
    AUX["aux BCE: direction\nover/under"]
    W["вес строки exp(-3·fdr)\nv2 only"]
    NPZ["train_all_features.npz\ntest_features.npz"]
    X --> NPZ
    NPZ --> CT --> HEAD
    W --> HEAD
    HEAD --> AUX
  end

  subgraph split [Разбиение]
    TR["train: все хромосомы кроме chr14"]
    VA["val: chr14"]
    TE["test: chr_test.h5 held-out"]
  end

  NPZ --> split
  HEAD --> OUT["pred comb_es + direction"]
```

**Поток данных по файлам:**

| Шаг | Скрипт | Вход | Выход |
|---|---|---|---|
| 1 | `extract_features_immune.py` | HDF5 + FASTA | `features_immune/*_features.npz` |
| 2 | `train_immune_v2.py` | NPZ + vocab JSON | `runs/effect_head_immune_v2/best.pkl` |
| 3 | `evaluate_immune_v2.py` | NPZ + checkpoint | метрики + PNG в `analysis_v2/` |

Опционально (эксперимент, как в курсовой «сырые эмбеддинги»):
`extract_embeddings_immune.py` → признак **3072-dim** =
`mean(embeddings_128bp_alt) − mean(embeddings_128bp_ref)` через
`build_embeddings_apply_fn` в `common.py`.

---

## Принципиальные отличия от старого кода (`AG/905_*`, `AG/1305_*`, `AG/emb_jax_immune.py`)

### 1. Учёт `cell_type` (главная ошибка старых попыток файнтюна)

В HDF5 **один и тот же SNP** (одна последовательность) встречается до **37 раз**
с **разным** `comb_es` — по одному на тип клетки.

| Старый код | Новый код |
|---|---|
| `905_finetune_full_immune.py`, `1305_new_finetune_all_immune.py` **не передают** `cell_type` в модель | `one_hot(cell_type)` **конкатенируется** с LFC-признаками |
| Один X → много разных Y без объяснения → **нерешаемый шум**, val corr **NaN** (см. логи) | Голова учит **разные веса по трекам** для B/T/NK/… |

### 2. Что именно подаётся в голову (сигнал ref vs alt)

| Подход | Признак | Проблема / плюс |
|---|---|---|
| Старый файнтюн: MSE на **средней** разности RNA-seq по окну / 128 bp | Размазанный сигнал SNP | Val ~0, NaN |
| `1305_new_finetune_all_immune.py` | `predict(..., resolutions=(128,))` | **TypeError** — скрипт не доходил до обучения |
| `emb_jax_immune.py` | mean по **всему** окну 128 kb, 667 треков, **без** cell_type в RF-пайплайне на диске | AUC direction на тех же фичах + cell_type ≈ **0.67** (мы перепроверили) |
| **Новый `CenterMaskScorer`** | **log2 sum alt/ref** в **2001 п.н.** вокруг варианта по RNA_SEQ трекам | Локальный ASE-сигнал, как в официальных рекомендациях AlphaGenome |
| Курсовой RF **0.763** | Описаны **сырые внутренние эмбеддинги** (скрипт в репо не сохранился); v2 на LFC **догнал** этот AUC (**0.766**) |

### 3. Обучение backbone vs замороженная модель + маленькая голова

Старые скрипты **размораживали** часть AlphaGenome и оптимизировали MSE end-to-end
на шумных таргетах → нестабильность. Новый пайплайн: **backbone frozen**,
обучается только **`EffectHead`** (~10⁴ параметров) на **кэшированных** признаках —
тот же принцип, что дал успех на PromoterAI (`best_code/`).

### 4. Исправления v2 относительно v1 (без смены признаков)

| v1 | v2 | Эффект |
|---|---|---|
| Равный вес всех строк | **Huber × exp(−3·fdr)** | Меньше шума от маломощных ASE |
| aux BCE на **`is_sig`** (FDR) | aux BCE на **direction** | AUC direction (FDR&lt;0.05): **0.69 → 0.77** |
| early stop по Pearson (all) | то же | Pearson на FDR&lt;0.05: **0.28 → 0.33** |

### 5. Инфраструктура (мелочи, но критичные на практике)

- **Координаты:** `position_start` в HDF5 — 0-based BED → **`+1`** для `genome.Variant`.
- **Дедупликация** вариантов при экстракции (~19843 уникальных вместо 81273 прогонов модели).
- **Val по chr14**, тест — отдельный **`chr_test.h5`** (не та же хромосома).
- **TensorBoard** в `train_immune.py` / `train_immune_v2.py`.
- **`load_alphagenome_model`**: локальный kagglehub-кэш без интерактивного login.

---

## Установка и запуск

Conda-env `ag`, зависимости — `requirements.txt` (как в `best_code/`).

```bash
export CUDA_VISIBLE_DEVICES=0
export FASTA=/path/to/hg38.fa
export CHR_TRAIN=/path/to/chr_train.h5
export CHR_TEST=/path/to/chr_test.h5

chmod +x run_immune_v2.sh
./run_immune_v2.sh
```

Или по шагам:

```bash
python extract_features_immune.py --h5 "$CHR_TRAIN" --split train_all \
  --fasta "$FASTA" --out-dir features_immune
python extract_features_immune.py --h5 "$CHR_TEST" --split test \
  --fasta "$FASTA" --out-dir features_immune \
  --vocab-path features_immune/cell_type_vocab.json

python train_immune_v2.py --features-dir features_immune \
  --out-dir runs/effect_head_immune_v2

python evaluate_immune_v2.py --features-dir features_immune \
  --checkpoint runs/effect_head_immune_v2/best.pkl \
  --plots-dir analysis_v2
```

TensorBoard:

```bash
tensorboard --logdir runs/effect_head_immune_v2/tensorboard --bind_all --port 6006
```

---

## Файлы в этой папке

| Файл | Назначение |
|---|---|
| `extract_features_immune.py` | LFC через `CenterMaskScorer` (основной пайплайн) |
| `extract_embeddings_immune.py` | эксперiment: сырые `embeddings_128bp` diff |
| `train_immune.py` | v1 (baseline, aux=is_sig) |
| `train_immune_v2.py` | **рекомендуемое** обучение |
| `evaluate_immune_v2.py` | метрики + графики direction |
| `evaluate_immune.py` | v1 evaluation |
| `heads.py`, `common.py` | общие модули |
| `run_immune.sh`, `run_immune_v2.sh` | оркестрация |
| `analysis/`, `analysis_v2/` | PNG с результатов прогона на сервере |
| `results_test_metrics.json` | val/test метрики (JSON) |
| `make_figures.py` | train/test figures |
| `report_test_metrics.py` | JSON metrics |
| `checkpoints/effect_head_immune_v*/best.pkl` | обученные чекпоинты (v1 baseline, **v2** основной) |

---

## Связь с `best_code/` (PromoterAI / SNP z-score)

Оба пайплайна используют одну идею: **не** учить backbone с нуля на
усреднённых эмбеддингах, а взять **официальный сигнал ref vs alt**
(`GeneMaskLFCScorer` для генов / `CenterMaskScorer` для ASE без gene_id) и
обучить **`EffectHead`**. Для иммунных данных добавлен обязательный
**`cell_type`**, без которого задача матемatically ill-posed.
