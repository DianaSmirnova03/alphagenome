"""Promoter: ref/alt one-hot + gene exon mask из CSV + hg38 + GTF (без npz-кэша seq)."""

from __future__ import annotations

import functools
from collections import OrderedDict

import numpy as np
import pandas as pd

from common import VariantRow, make_variant, make_window_interval, normalize_chrom, strip_ensembl_version


class PromoterSequenceLoader:
    def __init__(
        self,
        fasta_path: str,
        gtf_feather_path: str,
        *,
        window: int = 16384,
        cache_size: int = 8192,
    ):
        from alphagenome_research.io import fasta as ag_fasta
        from alphagenome_research.io import genome as genome_io
        from alphagenome_research.model import one_hot_encoder
        from alphagenome_research.model.variant_scoring import gene_mask_extractor as gme

        self.window = window
        self._extractor = ag_fasta.FastaExtractor(fasta_path)
        self._encoder = one_hot_encoder.DNAOneHotEncoder()
        self._genome_io = genome_io
        gtf = pd.read_feather(gtf_feather_path)
        self._mask_extractor = gme.GeneMaskExtractor(
            gtf=gtf,
            gene_mask_type=gme.GeneMaskType.EXONS,
            gene_query_type=gme.GeneQueryType.INTERVAL_CONTAINED,
        )
        self._cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_size = cache_size

    def _gene_idx(self, obs, gene_id, gene_name):
        if gene_id is not None and "gene_id" in obs.columns:
            obs_ids = obs["gene_id"].map(strip_ensembl_version).values
            idx = np.flatnonzero(obs_ids == strip_ensembl_version(gene_id))
            if len(idx) > 0:
                return int(idx[0])
        if gene_name is not None and "gene_name" in obs.columns:
            idx = np.flatnonzero(obs["gene_name"].values == gene_name)
            if len(idx) > 0:
                return int(idx[0])
        return None

    def get(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        *,
        gene_id: str | None = None,
        gene_name: str | None = None,
    ):
        key = (
            normalize_chrom(chrom),
            int(pos),
            str(ref).upper(),
            str(alt).upper(),
            strip_ensembl_version(gene_id),
            str(gene_name) if gene_name is not None else None,
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        vrow = VariantRow(chrom=key[0], pos=key[1], ref=key[2], alt=key[3], gene_id=gene_id, gene_name=gene_name)
        variant = make_variant(vrow)
        interval = make_window_interval(variant, self.window)
        ref_seq, alt_seq = self._genome_io.extract_variant_sequences(interval, variant, self._extractor)
        mask, obs = self._mask_extractor.extract(interval, variant)
        gene_idx = self._gene_idx(obs, gene_id, gene_name)
        if gene_idx is None:
            raise ValueError(f"gene not in mask: {key[0]}:{key[1]} gene_id={gene_id} gene={gene_name}")

        ref_oh = self._encoder.encode(ref_seq).astype(np.float32)
        alt_oh = self._encoder.encode(alt_seq).astype(np.float32)
        gene_mask = mask[:, gene_idx].astype(np.float32)
        out = (ref_oh, alt_oh, gene_mask)
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def stack_batch_from_df(self, df: pd.DataFrame, indices: np.ndarray):
        refs, alts, masks = [], [], []
        for i in indices:
            row = df.iloc[int(i)]
            gid = row["gene_id"] if "gene_id" in row.index and pd.notna(row.get("gene_id")) else None
            gname = row.get("gene") if "gene" in row.index else row.get("gene_name")
            r, a, m = self.get(
                row["chrom"],
                int(row["pos"]),
                str(row["ref"]),
                str(row["alt"]),
                gene_id=gid,
                gene_name=gname if pd.notna(gname) else None,
            )
            refs.append(r)
            alts.append(a)
            masks.append(m)
        return np.stack(refs), np.stack(alts), np.stack(masks)


def load_promoter_csv(csv_path: str, *, csv_sep: str = ",") -> pd.DataFrame:
    sep = "\t" if csv_sep == "\\t" else csv_sep
    return pd.read_csv(csv_path, sep=sep)


def filter_promoter_rows(df: pd.DataFrame, loader: PromoterSequenceLoader) -> np.ndarray:
    """Индексы строк CSV, для которых строится gene mask (как в sequence_dataset)."""
    ok = []
    for i in range(len(df)):
        row = df.iloc[i]
        gid = row["gene_id"] if "gene_id" in row.index and pd.notna(row.get("gene_id")) else None
        gname = row.get("gene") if "gene" in row.index else row.get("gene_name")
        try:
            loader.get(
                row["chrom"],
                int(row["pos"]),
                str(row["ref"]),
                str(row["alt"]),
                gene_id=gid,
                gene_name=gname if pd.notna(gname) else None,
            )
            ok.append(i)
        except (ValueError, Exception):
            continue
    return np.asarray(ok, dtype=np.int64)
