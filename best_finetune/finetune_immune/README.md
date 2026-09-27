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

### CenterMask LFC

В окне center mask \(M\) (длина **2001** на оси, выровнена к предсказаниям модели):

\[
S_{\mathrm{ref}}(t) = \sum_{i:\, M_i=1} \max(R^{\mathrm{ref}}_{i,t}, 0), \quad
S_{\mathrm{alt}}(t) = \sum_{i:\, M_i=1} \max(R^{\mathrm{alt}}_{i,t}, 0)
\]

\[
\mathrm{LFC}(t) = \log(S_{\mathrm{alt}}(t)+\varepsilon) - \log(S_{\mathrm{ref}}(t)+\varepsilon)
\]

Используются только треки с `valid_track_mask` из метаданных модели (как offline extract).

### Вход головы

\[
\mathbf{x} = \bigl[\,\tilde{\mathrm{LFC}}\,\|\,\mathrm{onehot}(\mathrm{cell\_type})\,\bigr],
\quad \dim = |\mathrm{valid\_tracks}| + N_{\mathrm{types}}+1
\]

### Loss v2

Регрессия с весами \(w_i = f(\mathrm{fdr\_comb\_pval}_i)\) (чем увереннее ASE, тем больше вес):

\[
L_z = \frac{\sum_i w_i \,\mathrm{Hubber}(\hat{z}_i - z_i)}{\sum_i w_i}
\]

Направление (если известно over/under):

\[
L_{\mathrm{aux}} = \mathrm{BCEWithLogits}(\mathrm{logit\_over}, \mathbb{1}[\mathrm{direction}=\mathrm{over}])
\]

\[
L = L_z + \lambda_{\mathrm{aux}} L_{\mathrm{aux}}, \quad \lambda_{\mathrm{aux}} = 0.5
\]

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

Артефакты: `../runs/e2e_head_immune_v2/best.pkl`, `analysis_v2/`.
