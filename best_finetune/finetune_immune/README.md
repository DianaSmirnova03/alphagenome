# Immune E2E: frozen AlphaGenome + CenterMask LFC + EffectHead v2

Этот каталог — **ASE / immune**: регрессия **comb_es** (и вспомогательная **direction**) для SNP
в контексте **типа клетки**. Данные — `chr_train.h5` + `hg38.fa`, **без** `features_immune/*_features.npz`
в training loop.

Общие модули — в [`../shared/`](../shared/).

---

## Что происходит по шагам

1. Строка h5: SNP (chr, pos, ref, alt), **cell_type**, таргет **comb_es** (`z`), FDR для весов.
2. Вырезается окно **16 384 bp**, строится **center mask** шириной **2001 bp** вокруг SNP.
3. Frozen AlphaGenome → RNA_SEQ ref/alt.
4. **CenterMask LFC** по валидным RNA-трекам → длинный вектор (как `extract_features_immune.py`).
5. К вектору **конкатенируется one-hot типа клетки** (37 типов + unknown).
6. EffectHead v2 → **ẑ** и logit **направления** (over vs under).
7. Loss = Huber(ẑ, z) с **FDR-весами** + **0.5 × BCE** по direction.

**Важно:** AlphaGenome **не** запускается отдельно «на каждый тип клетки». Один SNP в батче
может дать несколько строк h5 (разные `cell_type`) — forward ref/alt **дедуплицируется** по
(chr, pos, ref, alt), а one-hot говорит голове, **для какой клетки** нужен ответ.

Val — хромосома **14** (как Stage 2). Test — `chr_test.h5`.

---

## Формулы

> На GitHub: **`$$ … $$`** — блок формулы, **`$ … $`** — в тексте (не `\[` `\]`).

### CenterMask LFC

В окне center mask $M$ (ширина **2001** bp, выровнена к предсказаниям модели) для каждого трека $t$:

$$
S_{\text{ref}}(t) = \sum_{i:\, M_i = 1} \max(R^{\text{ref}}_{i,t},\, 0)
$$

$$
S_{\text{alt}}(t) = \sum_{i:\, M_i = 1} \max(R^{\text{alt}}_{i,t},\, 0)
$$

$$
\text{LFC}(t) = \log\bigl(S_{\text{alt}}(t)+\varepsilon\bigr) - \log\bigl(S_{\text{ref}}(t)+\varepsilon\bigr)
$$

Используются только треки с `valid_track_mask` из метаданных модели (как offline extract).

**Словами:** суммируем ref/alt RNA только в центральном окне вокруг SNP, затем log-ratio alt/ref.

### Вход головы

$$
\mathbf{x} = \bigl[\, \tilde{\text{LFC}} \;\|\; \text{onehot}(\text{cell\_type}) \,\bigr]
$$

$$
\dim(\mathbf{x}) = N_{\text{valid tracks}} + N_{\text{cell types}} + 1
$$

(last — unknown cell type).

### Loss v2

Регрессия с весами $w_i = f(\text{fdr\_comb\_pval}_i)$ (чем увереннее ASE, тем больше вес):

$$
L_z = \frac{\sum_i w_i \,\text{Huber}(\hat{z}_i - z_i)}{\sum_i w_i}
$$

Направление (over vs under):

$$
L_{\text{aux}} = \text{BCEWithLogits}(\text{logit\_over},\; \mathbb{1}[\text{direction}=\text{over}])
$$

$$
L = L_z + \lambda_{\text{aux}}\, L_{\text{aux}}, \quad \lambda_{\text{aux}} = 0.5
$$

---

## Параметры запуска (как в `launch_e2e_head_ft.sh`)

| Параметр | Значение |
|----------|----------|
| `--window` | **16384** |
| `--center-width` | **2001** |
| `--batch-size` | **2** |
| `--scaler-batch-size` | **2** |
| `--lr` | **1e-3** |
| `--weight-decay` | **1e-4** |
| `--max-epochs` | **200** |
| `--patience` | **20** |
| `--aux-weight` | **0.5** |
| `--fdr-weight-decay` | **3.0** |
| GPU | **CUDA_VISIBLE_DEVICES=0** |
| JAX VRAM | `PREALLOCATE=true`, `MEM_FRACTION=0.88` |
| Warm-start | `effect_head_immune_v2/best.pkl` |
| Train rows | ~**76776** (все хромосомы кроме 14) |
| Val | ~**4497** (chr14) |
| Test | `chr_test.h5` (~18k строк) |

**Почему batch 2, а у promoter 8:** один шаг immune держит **более длинный** LFC-вектор и
one-hot; на L40S batch 4 уже давал OOM при параллельной нагрузке на GPU 0.

Ожидаемое время: **несколько суток** (много шагов на эпоху + тяжёлый scaler-fit на ~77k строк).

---

## Инициализация головы

Как у promoter: веса MLP из Stage 2 [`best_code_immune`](../best_code_immune/), scaler по умолчанию
**пересчитывается** на train E2E. Это не использование test labels при обучении.

---

## Сравнение со старыми подходами

### Курсовой / ранний код

| Подход | Итог | Immune E2E |
|--------|------|------------|
| mean embedding | val r → 0, «не учится» | LFC center + cell type |
| LoRA immune без стабильного LFC | NaN / early stop | frozen AG + v2 loss |
| RF / только классификация | AUC ок, нет сквозной модели | регрессия + direction aux |

### Stage 2 [`best_code_immune`](../best_code_immune/)

| | Stage 2 npz | Immune E2E |
|---|-------------|------------|
| X | `extract_features_immune.py` | online CenterMask LFC |
| Loss | `train_immune_v2.py` — **тот же** `compute_loss` | тот же |
| val Pearson (ориентир) | **~0.33** | сравнивать после `best.pkl` |
| test Pearson all | **~0.19** | eval E2E |
| test dir AUC FDR&lt;0.05 | **~0.77** | eval E2E |

Promoter и immune **различаются кодом** (маска, данные, loss aux, batch), но **делят** `heads.py`
и `e2e_lfc.py` в `shared/`.

---

## Отличия от promoter E2E (кратко)

| | Promoter | Immune |
|---|----------|--------|
| Данные | CSV + GTF | h5 + vocab |
| LFC | GeneMask | CenterMask 2001 |
| Вход MLP | только LFC | LFC + one-hot cell |
| Loss | Huber(z) | Huber(z) + BCE(direction) |
| batch | 8 | 2 |
| GPU в launch | 3 | 0 |

---

## Результаты E2E (held-out test + val)

Сводка: [`results_test_metrics.json`](results_test_metrics.json).  
Графики: [`analysis/`](analysis/) — `make_figures.py --e2e`.  
Eval: `evaluate_e2e_immune.py` (online CenterMask LFC + one-hot cell type).

### Таблица метрик

| Метрика | **E2E test/val** | Stage 2 [`best_code_immune`](../best_code_immune/) |
|---------|------------------|------------------------------------------------------|
| Val Pearson r (chr14, n=4497) | **0.171** | **0.329** |
| Train best val r (early stop) | **0.175** | — |
| Test Pearson r (all) | **0.136** | **0.187** |
| Test Pearson r (FDR &lt; 0.05) | **0.290** | **0.331** |
| Direction AUC (all) | **0.582** | **0.608** |
| Direction AUC (FDR &lt; 0.05) | **0.675** | **0.766** |

**Вывод:** E2E immune **слабее** Stage 2 на npz, особенно val и direction AUC; на FDR&lt;0.05 регрессия ближе к Stage 2 (~0.29 vs ~0.33).

---

## Графики и выводы (`analysis/`)

<table>
<tr>
<td width="33%" valign="top"><img src="analysis/train_curve.png" alt="train_curve" width="100%"/><p><strong>E2E обучение v2 loss</strong></p><p>Train loss (Huber + FDR-веса + direction aux) и val Pearson на chr14. Early stop ~эпоха 46, best val r ≈ **0.175**.</p><p>Ниже плато Stage 2 (~0.33): online LFC и малый batch=2 усложняют fit; val шумный.</p></td>
<td width="33%" valign="top"><img src="analysis/val_scatter_immune_v2.png" alt="val_scatter" width="100%"/><p><strong>Val chr14 (E2E forward)</strong></p><p><code>comb_es</code> vs pred; 4497 строк, все cell types; pred через GPU forward, не npz.</p><p>r ≈ **0.17** на E2E eval — ниже Stage 2 scatter (~0.33).</p></td>
<td width="33%" valign="top"><img src="analysis/test_roc_direction_v2.png" alt="test_roc" width="100%"/><p><strong>ROC direction</strong></p><p>Over vs under; score = pred. All test и FDR&lt;0.05.</p><p>AUC FDR&lt;0.05 ≈ **0.67** vs **0.77** у Stage 2 — знак ловится слабее, но лучше случайного.</p></td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis/test_pr_direction_fdr005.png" alt="test_pr" width="100%"/><p><strong>PR direction (FDR&lt;0.05)</strong></p><p>Precision–recall на достоверных ASE.</p><p>Согласуется с ROC ~0.67; operating point для отбора вариантов.</p></td>
<td width="33%" valign="top"><img src="analysis/test_scatter_all.png" alt="scatter_all" width="100%"/><p><strong>Test scatter (all)</strong></p><p>18766 строк <code>chr_test.h5</code>; r ≈ **0.14**.</p><p>Низкий r на all ожидаем (шумные метки); сравнимо со Stage 2 ~0.19, чуть ниже.</p></td>
<td width="33%" valign="top"><img src="analysis/test_scatter_fdr005.png" alt="scatter_fdr" width="100%"/><p><strong>Test FDR&lt;0.05</strong></p><p>Только надёжные ASE; r ≈ **0.29**.</p><p>Ближе к Stage 2 (~0.33): модель регрессирует модуль там, где метка доверена.</p></td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis/test_pearson_by_cell_type_v2.png" alt="pearson_ct" width="100%"/><p><strong>Pearson по cell type</strong></p><p>r на test отдельно по типам клеток (n≥10).</p><p>Гeterogeneity сохраняется; редкие типы — нестабильные столбцы.</p></td>
<td width="33%" valign="top"><img src="analysis/test_heatmap_cell_type.png" alt="heatmap" width="100%"/><p><strong>Heatmap r и AUC</strong></p><p>Две метрики по cell types на test.</p><p>Часто AUC direction выше r: знак ловится там, где модуль comb_es шумный.</p></td>
<td width="33%" valign="top"><img src="analysis/test_hist_abs_fdr005.png" alt="hist" width="100%"/><p><strong>|comb_es| vs |pred|</strong></p><p>На FDR&lt;0.05 — сжатие хвостов (regression to mean).</p><p>Как у Stage 2: Huber + MLP не повторяет экстремальные effect size.</p></td>
</tr>
</table>

---

## Файлы в этой папке

| Файл | Роль |
|------|------|
| `finetune_e2e_head_immune.py` | Обучение |
| `evaluate_e2e_immune.py` | Val/test forward |
| `immune_sequence_loader.py` | FASTA, center mask |
| `e2e_batch_utils.py` | дедуп SNP в батче |
| `extract_features_immune.py` | `_load_h5` |
| `train_immune.py`, `train_immune_v2.py` | батчи, FDR, direction AUC |

## Запуск только immune

```bash
cd .. && bash launch_e2e_head_ft.sh immune
```

Артефакты: `../runs/e2e_head_immune_v2/best.pkl`, графики [`analysis/`](analysis/).
