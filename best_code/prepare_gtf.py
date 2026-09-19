"""Конвертирует GENCODE GTF в `.feather`-таблицу нужного формата для
`alphagenome_research.model.variant_scoring.gene_mask_extractor.GeneMaskExtractor`.

Требуемые колонки (см. `gene_mask_extractor.py`, `_GeneExonAnnotationExtractor`
и `_GeneBodyAnnotationExtractor`): `Chromosome, Start, End, Strand, Feature,
gene_id, gene_name, gene_type, transcript_id`, координаты 0-based
полуоткрытые (стандарт PyRanges). Это ровно то, что нужно официальному
`AlphaGenomeModel` при `organism_settings[...].gtf_feather_path` — по
умолчанию модель тянет уже готовый `gencode.v46...gtf.gz.feather` с GCS, а
у пользователя локально лежит `gencode.v39.annotation.gtf`, поэтому нужен
локальный конвертер, использующий тот же pyranges-подход, что и
`my_data/my/linear_head_masked/prepare_data.py`.

Запуск:
    python prepare_gtf.py --gtf /path/to/gencode.v39.annotation.gtf \
        --out gencode.v39.feather
"""

from __future__ import annotations

import argparse

import pandas as pd


REQUIRED_COLUMNS = [
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "Feature",
    "gene_id",
    "gene_name",
    "gene_type",
    "transcript_id",
]


def convert(gtf_path: str, out_path: str) -> None:
    import pyranges as pr

    gr = pr.read_gtf(gtf_path)
    df = gr.df

    # Некоторые версии GENCODE называют биотип гена `gene_biotype` вместо
    # `gene_type` (например, при экспорте через Ensembl). Приводим к схеме,
    # которую ожидает GeneMaskExtractor.
    if "gene_type" not in df.columns and "gene_biotype" in df.columns:
        df["gene_type"] = df["gene_biotype"]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В GTF не хватает колонок {missing} после парсинга pyranges. "
            f"Доступные колонки: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]]

    # Chromosome должен быть с префиксом chr, как в hg38.fa/ FastaExtractor.
    df["Chromosome"] = df["Chromosome"].astype(str)
    needs_prefix = ~df["Chromosome"].str.startswith("chr")
    df.loc[needs_prefix, "Chromosome"] = "chr" + df.loc[needs_prefix, "Chromosome"]

    df.reset_index(drop=True).to_feather(out_path)
    print(
        f"Записано {len(df)} строк ({df.Feature.value_counts().to_dict()}) в"
        f" {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", required=True, help="Путь к .gtf/.gtf.gz")
    parser.add_argument("--out", required=True, help="Путь к выходному .feather")
    args = parser.parse_args()
    convert(args.gtf, args.out)


if __name__ == "__main__":
    main()
