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
<td width="33%" valign="top"><img src="analysis/train_curve.png" alt="train_curve" width="100%"/><p><strong>Кривая обучения</strong></p><p>На графике по горизонтали — номер эпохи. Синяя линия — средний train loss (Huber по таргету <code>z</code>); оранжевая — Pearson r между предсказанием EffectHead и истинным <code>z</code> на валидации (2044 варианта). Красная точка отмечает эпоху с максимальным val r — по ней сохранён чекпоинт <code>best.pkl</code>.</p><p>Loss монотонно снижается без «взрыва», характерного для старых LoRA-запусков. Val r выходит на плато примерно 0.26–0.29 и не обнуляется к концу обучения, в отличие от линейных голов на усреднённых эмбеддингах, где корреляция сначала поднимается до ~0.2, а затем деградирует. Это главный признак того, что голова учится на осмысленном LFC-сигнале, а не подстраивается под шум.</p></td>
<td width="33%" valign="top"><img src="analysis/val_scatter.png" alt="val_scatter" width="100%"/><p><strong>Val: истина vs предсказание</strong></p><p>Каждая точка — один SNP из <code>val_features.npz</code>: по оси X истинный z-score эффекта на экспрессию, по Y — предсказание обученной головы после forward через <code>best.pkl</code>. На рисунке обычно указан Pearson r (~0.29).</p><p>Облако не выглядит как случайный шум: виден наклон «больше z → выше pred», но разброс большой. Это ожидаемо: часть вариации в <code>z</code> связана с контекстом эксперимента и геномной архитектурой, которую одна замена в промоторе не полностью кодирует. Тем не менее связь стабильнее и сильнее, чем у всех прежних попыток файнтюнинга на raw embeddings.</p></td>
<td width="33%" valign="top"><img src="analysis/val_residual.png" alt="val_residual" width="100%"/><p><strong>Остатки на val</strong></p><p>По оси X — истинный <code>z</code>, по Y — ошибка pred − <code>z</code> для тех же вариантов, что на scatter. Идеальная модель дала бы горизонтальную полосу вокруг нуля без систематической формы.</p><p>Нет выраженной U-образной или воронкообразной структуры: модель не систематически недооценивает экстремальные |z| и не «залипает» в предсказании около нуля. Остатки распределены относительно симметрично, что согласуется с регрессией без грубого смещения по величине эффекта.</p></td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis/test_roc.png" alt="test_roc" width="100%"/><p><strong>ROC на held-out test</strong></p><p>Held-out набор <code>tableS1A</code> (749 вариантов): числового <code>z</code> нет, только класс <code>consequence</code> (over / under / none). Score для ROC — предсказанный ẑ. Три кривые: over против none, under против none, over против under — та же постановка, что в <code>AG/rf.py</code>.</p><p>Все три AUC выше 0.77; сильнее всего разделение over vs under (≈0.91). Модель не только отличает «есть регуляторный эффект» от none, но и хорошо разводит знаки over и under — практически полезная задача для интерпретации allelic imbalance.</p></td>
<td width="33%" valign="top"><img src="analysis/test_pr.png" alt="test_pr" width="100%"/><p><strong>Precision–Recall на test</strong></p><p>Те же три pairwise задачи, что на ROC, но в координатах precision и recall. На test классы none / over / under несбалансированы, поэтому одной ROC недостаточно, чтобы понять качество на редких positive.</p><p>Кривые показывают, при каком пороге по ẑ можно получить высокий recall для over (или under) и какой ценой падает precision. Для over vs none видно, что агрессивный отбор по score даёт много найденных over при умеренной доле ложных срабатываний — согласуется с высоким AUC и с KDE по классам.</p></td>
<td width="33%" valign="top"><img src="analysis/test_kde.png" alt="test_kde" width="100%"/><p><strong>Распределение ẑ по классам</strong></p><p>На test для каждого варианта известен только <code>consequence</code>. На графике — оценки плотности (KDE) предсказанного ẑ отдельно для none, over и under.</p><p>Распределение none сосредоточено около нуля. Over смещён в положительную область, under — в отрицательную, с частичным перекрытием у нуля. Это значит, что голова кодирует не бинарное «эффект / нет», а непрерывный score, согласованный со знаком allelic effect — то, что нужно и для регрессии на val, и для классификации на test.</p></td>
</tr>
<tr>
<td width="33%" valign="top"><img src="analysis/test_calibration_over.png" alt="test_calibration_over" width="100%"/><p><strong>Калибровка для класса over</strong></p><p>Варианты test отсортированы по предсказанному ẑ и разбиты на 10 decile-бинов. По X — средний ẑ в бине, по Y — доля истинных over среди всех вариантов в этом бине.</p><p>Тренд монотонный: чем выше decile по ẑ, тем чаще в бине встречается over. Это грубая, но наглядная калибровка: score можно трактовать как «уверенность в направлении over», без отдельного Platt scaling. Для under аналогичная картина ожидается в отрицательных decile (см. KDE).</p></td>
<td width="33%" valign="top"><img src="analysis/test_auc_rf_vs_head.png" alt="test_auc_rf_vs_head" width="100%"/><p><strong>EffectHead vs Random Forest</strong></p><p>Столбчатое сравнение AUC на том же test и тех же LFC-признаках. Для RF показан размах min–max по трём независимым запускам из курсового <code>rf.py</code>; для EffectHead — одно значение с текущего <code>best.pkl</code>.</p><p>EffectHead не проигрывает RF ни по одной из трёх пар; по over/under AUC даже выше верхней границы RF. При этом голова — компактная дифференцируемая сеть, которую можно стыковать со Stage 3 (разморозка последнего слоя RNA_SEQ), тогда как RF остаётся отдельным black-box классификатором на тех же фичах.</p></td>
<td width="33%" valign="top"><img src="analysis/metrics_card.png" alt="metrics_card" width="100%"/><p><strong>Сводка метрик</strong></p><p>Текстовая «открытка» с ключевыми числами: val Pearson r, test AUC для трёх pairwise сравнений, размеры выборок — те же значения, что в <code>results_test_metrics.json</code> и таблице выше в README.</p><p>Удобна для быстрой сверки после перегенерации графиков: если JSON обновился, карточка и таблица в README должны совпадать. Сама по себе не добавляет новой аналитики, но фиксирует итог пайплайна в одном кадре для отчёта или слайда.</p></td>
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
