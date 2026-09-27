"""Общие утилиты: загрузка предобученной модели AlphaGenome, построение
genome.Variant/Interval, сидирование.

Всё, что касается самой модели AlphaGenome, максимально переиспользует
официальный код из пакета `alphagenome_research` (тот же пакет, которым
пользовались рабочие скрипты `AG/emb_jax_2304.py`, `AG/rf.py`) и `alphagenome`
(`alphagenome.data.genome`, `alphagenome.models.dna_output`). Никакой логики
модели здесь не переизобретается.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclasses.dataclass(frozen=True)
class VariantRow:
    """Единообразное представление одной строки CSV с вариантом."""

    chrom: str
    pos: int
    ref: str
    alt: str
    gene_id: Optional[str] = None
    gene_name: Optional[str] = None
    z: Optional[float] = None
    p_over: Optional[float] = None
    p_under: Optional[float] = None
    consequence: Optional[str] = None


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def strip_ensembl_version(gene_id: Optional[str]) -> Optional[str]:
    """Убирает версию из Ensembl ID (`ENSG00000134121.10` -> `ENSG00000134121`).

    Важно: GENCODE GTF всегда хранит `gene_id` с версией (`.10` и т.п.), а
    пользовательские CSV (`train_variants*.csv`, `tableS1A.tsv`) — без неё.
    Без этой нормализации сопоставление по `gene_id` тихо ничего не находит
    (проверено на реальных данных: `ENSG00000134121` из CSV не совпадает с
    `ENSG00000134121.10` из GTF), и все примеры молча пропускаются.
    """

    if gene_id is None:
        return None
    return str(gene_id).split(".")[0]


def make_variant(row: VariantRow):
    """Строит `alphagenome.data.genome.Variant` из строки CSV.

    `genome.Variant.position` — 1-based (см. docstring класса), т.е. ровно то,
    что лежит в колонке `pos` стандартного VCF-подобного CSV. Старый код
    (`my_data/my/*/data_loader.py`) использовал `pos` как 0-based координату
    в FASTA напрямую, что могло сдвигать окно на 1 п.н. Переиспользуя
    `genome.Variant`/`genome.Interval`, мы получаем ту же (проверенную)
    логику перевода координат, что и во всех predict_variant/score_variant
    вызовах AlphaGenome.
    """

    from alphagenome.data import genome

    return genome.Variant(
        chromosome=normalize_chrom(row.chrom),
        position=int(row.pos),
        reference_bases=str(row.ref).upper(),
        alternate_bases=str(row.alt).upper(),
    )


def make_window_interval(variant, window: int):
    """Возвращает `genome.Interval` шириной `window`, отцентрированный на
    варианте (через `variant.reference_interval.resize`, официальный метод
    `alphagenome.data.genome.Interval`)."""

    return variant.reference_interval.resize(window)


def _find_cached_kaggle_checkpoint(model_version: str) -> Optional[str]:
    """Ищет уже скачанный kagglehub-чекпойнт локально, чтобы не дёргать
    `kagglehub.login()` (интерактивный промпт, ломается в неинтерактивном
    режиме) при каждом запуске, если веса уже есть на диске."""

    import glob
    import os

    base = os.path.expanduser(
        f"~/.cache/kagglehub/models/google/alphagenome/jax/{model_version.lower()}"
    )
    if not os.path.isdir(base):
        return None
    # Структура: .../<model_version>/<version_number>/ (например .../all_folds/1/)
    candidates = sorted(glob.glob(os.path.join(base, "*")))
    candidates = [c for c in candidates if os.path.isdir(c)]
    return candidates[-1] if candidates else None


def load_alphagenome_model(
    model_version: str = "all_folds",
    fasta_path: Optional[str] = None,
    gtf_feather_path: Optional[str] = None,
    device=None,
):
    """Загружает предобученную модель AlphaGenome.

    Сначала пытается использовать уже скачанный локально kagglehub-чекпойнт
    (см. `_find_cached_kaggle_checkpoint`) через `dna_model.create(...)` —
    это тот же путь, который использует `dna_model.create_from_kaggle`
    внутри, но без интерактивного `kagglehub.login()`, если веса уже
    закэшированы (что верно для этой машины: `~/.cache/kagglehub/models/
    google/alphagenome/...` уже заполнен предыдущими запусками
    `AG/emb_jax_2304.py`/`AG/rf.py`). Если кэша нет — падает обратно на
    `create_from_kaggle`, как и раньше (тогда понадобится Kaggle-логин).
    """

    from alphagenome.models.dna_model import Organism
    from alphagenome_research.model import dna_model

    organism_settings = None
    if fasta_path is not None or gtf_feather_path is not None:
        organism_settings = {
            Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
                fasta_path=fasta_path,
                gtf_feather_path=gtf_feather_path,
            )
        }

    cached_checkpoint = _find_cached_kaggle_checkpoint(model_version)
    if cached_checkpoint is not None:
        return dna_model.create(
            cached_checkpoint,
            organism_settings=organism_settings,
            device=device,
        )

    return dna_model.create_from_kaggle(
        model_version,
        organism_settings=organism_settings,
        device=device,
    )


def build_differentiable_apply_fn(model):
    """Возвращает дифференцируемую (через `jax.grad`) функцию форварда модели.

    `dna_model.create_model` строит те же `hk.transform_with_state`-функции,
    что использовались при предобучении AlphaGenome (см.
    `alphagenome_research/model/dna_model.py::create_model`). В отличие от
    `model.predict_variant`/`model.score_variant` (которые под капотом делают
    `jax.device_get`, то есть возвращают numpy и рвут граф автодифференцирования),
    `apply_fn` из `create_model` — чистая функция
    `(params, state, sequence[B,S,4], organism_index[B]) -> predictions`,
    пригодная для `jax.value_and_grad`. Используется только в
    `finetune_backbone.py` (Stage 3).
    """

    from alphagenome_research.model import dna_model

    # model._metadata — приватное поле AlphaGenomeModel, но именно так же его
    # читает существующий код пользователя (`model._params`, `model._metadata`
    # в старых скриптах `AG/*`), так что это не новый паттерн, а
    # единообразный с уже работающим кодом способ получить нужный формат
    # metadata для create_model.
    init_fn, apply_fn, junctions_apply_fn = dna_model.create_model(model._metadata)
    return init_fn, apply_fn, junctions_apply_fn


def build_embeddings_apply_fn(model):
    """Возвращает `apply_fn`, отдающий сырые внутренние эмбеддинги
    `embeddings_128bp` (выход `TransformerTower`, [B, S//128, 3072]) вместо
    предсказаний голов.

    Используется только для эксперимента "признаки = разность сырых
    внутренних эмбеддингов ref/alt" для иммунных ASE-данных (см.
    `extract_embeddings_immune.py`) — попытка воспроизвести рецепт из
    курсовой (RF на эмбеддингах AlphaGenome, AUC 0.763), который не
    сохранился как отдельный скрипт в репозитории пользователя.

    `dna_model.create_model` (см. `build_differentiable_apply_fn`) отдаёт
    только `predictions['embeddings_1bp']` — 1536-дим представление НА
    КАЖДУЙ П.Н. (для окна 131072 это 131072*1536*4 байта ~ 805 МБ на один
    пример, непрактично для мини-батчей). `embeddings_128bp` — то же самое
    представление до `SequenceDecoder`, агрегированное на разрешении 128
    п.н. (3072-дим на каждые 128 п.н., то есть в 128 раз компактнее) — тот
    же widely-used в оригинальном AlphaGenome сигнал "глобального контекста"
    (см. docstring `AlphaGenome.__call__` в `alphagenome_research/model/
    model.py`), просто не проброшенный наружу в стандартном `create_model`.
    """

    import haiku as hk
    import jmp
    from alphagenome_research.model import model as ag_model_module

    metadata = model._metadata  # noqa: SLF001 — см. build_differentiable_apply_fn
    jmp_policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(ag_model_module.AlphaGenome, jmp_policy):
            return ag_model_module.AlphaGenome(metadata)(dna_sequence, organism_index)

    def embeddings_apply_fn(params, state, dna_sequence, organism_index):
        (_, embeddings), _ = _forward.apply(params, state, None, dna_sequence, organism_index)
        return embeddings.embeddings_128bp

    return embeddings_apply_fn


def get_rna_seq_valid_mask(model) -> np.ndarray:
    """Булева маска непаддинговых RNA_SEQ треков для человека."""

    from alphagenome.models.dna_model import Organism
    from alphagenome.models import dna_output

    metadata = model._metadata[Organism.HOMO_SAPIENS]
    return ~np.asarray(metadata.padding[dna_output.OutputType.RNA_SEQ])
