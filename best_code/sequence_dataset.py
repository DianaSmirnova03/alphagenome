"""Кэширование one-hot ref/alt последовательностей + маски гена для Stage 3
(`finetune_backbone.py`).

В отличие от `extract_features.py` (который использует официальный
`model.score_variant`, не дифференцируемый), здесь нужны сырые входы модели
(`sequences[B,S,4]`) и маска гена, чтобы прогонять их через дифференцируемый
`apply_fn` напрямую. Переиспользуется:
  - `alphagenome_research.model.one_hot_encoder.DNAOneHotEncoder`
  - `alphagenome_research.io.genome.extract_variant_sequences` (корректно
    обрабатывает вставку REF/ALT и strand, в отличие от ручного кода в
    `my_data/my/*/data_loader.py`)
  - `alphagenome_research.model.variant_scoring.gene_mask_extractor.GeneMaskExtractor`
    (GeneMaskType.EXONS) — то же самое, что использует официальный
    `GeneMaskLFCScorer`.

Внимание: при большом `--window` кэш может быть большим (one-hot uint8:
window * 4 байта на последовательность, ref+alt на пример). Для Stage 3
рекомендуется window <= 16384 и умеренное число примеров (см. README).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import VariantRow, make_variant, make_window_interval, set_seed, strip_ensembl_version


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


def build_dataset(
    csv_path: str,
    split: str,
    fasta_path: str,
    gtf_feather_path: str,
    out_dir: str,
    window: int = 16384,
    csv_sep: str = ",",
    max_examples: Optional[int] = None,
) -> None:
    from alphagenome_research.io import fasta as ag_fasta
    from alphagenome_research.io import genome as genome_io
    from alphagenome_research.model import one_hot_encoder
    from alphagenome_research.model.variant_scoring import gene_mask_extractor as gme

    set_seed(0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=csv_sep)
    if max_examples is not None:
        df = df.sample(n=min(max_examples, len(df)), random_state=0).reset_index(drop=True)

    extractor = ag_fasta.FastaExtractor(fasta_path)
    encoder = one_hot_encoder.DNAOneHotEncoder()
    gtf = pd.read_feather(gtf_feather_path)
    mask_extractor = gme.GeneMaskExtractor(
        gtf=gtf,
        gene_mask_type=gme.GeneMaskType.EXONS,
        gene_query_type=gme.GeneQueryType.INTERVAL_CONTAINED,
    )

    ref_onehot, alt_onehot, gene_masks, kept_rows = [], [], [], []
    n_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"seq_dataset[{split}]"):
        vrow = _row_to_variant_row(row)
        try:
            variant = make_variant(vrow)
            interval = make_window_interval(variant, window)
            ref_seq, alt_seq = genome_io.extract_variant_sequences(interval, variant, extractor)

            mask, obs = mask_extractor.extract(interval, variant)
            gene_idx = None
            if vrow.gene_id is not None and "gene_id" in obs.columns:
                # GTF хранит gene_id с версией (`ENSG...10`), CSV — без неё.
                obs_ids = obs["gene_id"].map(strip_ensembl_version).values
                idx = np.flatnonzero(obs_ids == strip_ensembl_version(vrow.gene_id))
                if len(idx) > 0:
                    gene_idx = int(idx[0])
            if gene_idx is None and vrow.gene_name is not None and "gene_name" in obs.columns:
                idx = np.flatnonzero(obs["gene_name"].values == vrow.gene_name)
                if len(idx) > 0:
                    gene_idx = int(idx[0])
            if gene_idx is None:
                n_skipped += 1
                continue

            ref_onehot.append(encoder.encode(ref_seq).astype(np.uint8))
            alt_onehot.append(encoder.encode(alt_seq).astype(np.uint8))
            gene_masks.append(mask[:, gene_idx].astype(bool))
            kept_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            n_skipped += 1
            tqdm.write(f"[warn] пропущен {vrow.chrom}:{vrow.pos} ({exc})")
            continue

    print(f"Готово: {len(kept_rows)} успешно, {n_skipped} пропущено.")
    if not kept_rows:
        raise RuntimeError("Ни одного примера не удалось обработать.")

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    np.savez(
        out_dir / f"{split}_sequences.npz",
        ref_onehot=np.stack(ref_onehot, axis=0),
        alt_onehot=np.stack(alt_onehot, axis=0),
        gene_mask=np.stack(gene_masks, axis=0),
        z=kept_df["z"].to_numpy(dtype=np.float32)
        if "z" in kept_df.columns
        else np.full(len(kept_df), np.nan, dtype=np.float32),
    )
    print(f"Сохранено {out_dir / f'{split}_sequences.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--gtf-feather", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=int, default=16384)
    parser.add_argument("--csv-sep", default=",")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    csv_sep = "\t" if args.csv_sep == "\\t" else args.csv_sep

    build_dataset(
        csv_path=args.csv,
        split=args.split,
        fasta_path=args.fasta,
        gtf_feather_path=args.gtf_feather,
        out_dir=args.out_dir,
        window=args.window,
        csv_sep=csv_sep,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
