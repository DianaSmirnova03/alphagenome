"""Извлечение признаков эффекта варианта: официальный GENE_MASK_LFC.

Для каждого варианта из CSV вызывает
`model.score_variant(interval, variant,
variant_scorers=[GeneMaskLFCScorer(requested_output=OutputType.RNA_SEQ)])`
— тот же метод, который лежит в основе рабочего пайплайна
`AG/rf.py`/`AG/emb_jax_2304.py` (там use `predict_variant` +
`(alt - ref).mean(axis=0)` вручную; здесь используется чуть более точная,
полностью официальная версия той же идеи — `GeneMaskLFCScorer`, которая
дополнительно учитывает точную маску экзонов нужного гена и его strand, а не
усредняет по всему окну без разбора).

Результат — вектор `log(alt_mean/ref_mean)` по всем непаддинговым RNA_SEQ
трекам, посчитанный только по позициям экзонов гена из колонки `gene_id`
(`gene` для tableS1A). Это прямая мера того, как сильно вариант меняет
предсказанную экспрессию именно в интересующем гене — то есть модель уже
здесь "видит" разницу ref/alt, до какой-либо обучаемой головы.

Запуск:
    python extract_features.py --csv AG/data/train_variants1.csv --split train \
        --fasta /path/hg38.fa --gtf-feather gencode.v39.feather --out-dir features
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    VariantRow,
    load_alphagenome_model,
    make_variant,
    make_window_interval,
    set_seed,
    strip_ensembl_version,
)


CONSEQUENCE_TO_LABEL = {"none": 0, "over": 1, "under": 2}


def _row_to_variant_row(row: pd.Series) -> VariantRow:
    return VariantRow(
        chrom=row["chrom"],
        pos=row["pos"],
        ref=row["ref"],
        alt=row["alt"],
        gene_id=row.get("gene_id"),
        gene_name=row.get("gene"),
        z=row.get("z"),
        p_over=row.get("p_over"),
        p_under=row.get("p_under"),
        consequence=row.get("consequence"),
    )


def _find_gene_row_index(obs: pd.DataFrame, gene_id: Optional[str], gene_name: Optional[str]):
    if gene_id is not None and "gene_id" in obs.columns:
        # GTF хранит gene_id с версией (`ENSG...10`), CSV — без неё.
        obs_ids = obs["gene_id"].map(strip_ensembl_version).values
        idx = np.flatnonzero(obs_ids == strip_ensembl_version(gene_id))
        if len(idx) > 0:
            return int(idx[0])
    if gene_name is not None and "gene_name" in obs.columns:
        idx = np.flatnonzero(obs["gene_name"].values == gene_name)
        if len(idx) > 0:
            return int(idx[0])
    return None


def extract_features(
    csv_path: str,
    split: str,
    fasta_path: str,
    gtf_feather_path: str,
    out_dir: str,
    window: int = 131072,
    csv_sep: str = ",",
    model_version: str = "all_folds",
    limit: Optional[int] = None,
) -> None:
    from alphagenome.models import dna_output
    from alphagenome.models import variant_scorers

    set_seed(0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=csv_sep)
    if limit is not None:
        df = df.iloc[:limit]

    print(f"Загружено {len(df)} строк из {csv_path}")
    model = load_alphagenome_model(
        model_version=model_version,
        fasta_path=fasta_path,
        gtf_feather_path=gtf_feather_path,
    )
    scorer_settings = variant_scorers.GeneMaskLFCScorer(
        requested_output=dna_output.OutputType.RNA_SEQ
    )

    features = []
    kept_rows = []
    n_skipped_gene_not_found = 0
    n_skipped_error = 0
    track_metadata_saved = False

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"extract[{split}]"):
        vrow = _row_to_variant_row(row)
        try:
            variant = make_variant(vrow)
            interval = make_window_interval(variant, window)
            results = model.score_variant(
                interval, variant, variant_scorers=[scorer_settings]
            )
            ann = results[0]
        except Exception as exc:  # noqa: BLE001 - логируем и продолжаем
            n_skipped_error += 1
            tqdm.write(f"[warn] пропущен вариант {vrow.chrom}:{vrow.pos} ({exc})")
            continue

        gene_idx = _find_gene_row_index(ann.obs, vrow.gene_id, vrow.gene_name)
        if gene_idx is None:
            n_skipped_gene_not_found += 1
            continue

        features.append(np.asarray(ann.X[gene_idx, :], dtype=np.float32))
        kept_rows.append(row)

        if not track_metadata_saved:
            ann.var.reset_index(drop=True).to_csv(out_dir / "track_metadata.csv", index=False)
            track_metadata_saved = True

    print(
        f"Готово: {len(features)} успешно, {n_skipped_gene_not_found} пропущено"
        f" (ген не найден в окне {window} п.н. — попробуйте увеличить --window),"
        f" {n_skipped_error} пропущено из-за ошибок."
    )

    if not features:
        raise RuntimeError("Ни одного варианта не удалось обработать — проверьте пути к fasta/gtf.")

    X = np.stack(features, axis=0)
    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)

    z = kept_df["z"].to_numpy(dtype=np.float32) if "z" in kept_df.columns else None
    p_over = kept_df["p_over"].to_numpy(dtype=np.float32) if "p_over" in kept_df.columns else None
    p_under = kept_df["p_under"].to_numpy(dtype=np.float32) if "p_under" in kept_df.columns else None
    consequence = (
        kept_df["consequence"].map(CONSEQUENCE_TO_LABEL).to_numpy(dtype=np.int64)
        if "consequence" in kept_df.columns
        else None
    )

    np.savez(
        out_dir / f"{split}_features.npz",
        X=X,
        z=z if z is not None else np.full(len(X), np.nan, dtype=np.float32),
        p_over=p_over if p_over is not None else np.full(len(X), np.nan, dtype=np.float32),
        p_under=p_under if p_under is not None else np.full(len(X), np.nan, dtype=np.float32),
        consequence=consequence if consequence is not None else np.full(len(X), -1, dtype=np.int64),
    )
    kept_df.to_csv(out_dir / f"{split}_meta.csv", index=False)
    print(f"Сохранено: {out_dir / f'{split}_features.npz'} (X shape={X.shape})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--gtf-feather", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=int, default=131072)
    parser.add_argument("--csv-sep", default=",")
    parser.add_argument("--model-version", default="all_folds")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число строк (для быстрого теста)")
    args = parser.parse_args()

    csv_sep = args.csv_sep
    if csv_sep == "\\t":
        csv_sep = "\t"

    extract_features(
        csv_path=args.csv,
        split=args.split,
        fasta_path=args.fasta,
        gtf_feather_path=args.gtf_feather,
        out_dir=args.out_dir,
        window=args.window,
        csv_sep=csv_sep,
        model_version=args.model_version,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
