"""On-the-fly ref/alt one-hot + center mask для иммунных вариантов (без гигабайтного кэша)."""

from __future__ import annotations

import functools
from collections import OrderedDict

import numpy as np

from common import VariantRow, make_variant, make_window_interval, normalize_chrom


class ImmuneSequenceLoader:
    def __init__(
        self,
        fasta_path: str,
        *,
        window: int = 16384,
        center_width: int = 2001,
        cache_size: int = 4096,
    ):
        from alphagenome.models import dna_output
        from alphagenome_research.io import fasta as ag_fasta
        from alphagenome_research.io import genome as genome_io
        from alphagenome_research.model import one_hot_encoder
        from alphagenome_research.model.variant_scoring import center_mask as cm
        from alphagenome_research.model.variant_scoring import variant_scoring

        self.window = window
        self.center_width = center_width
        self._extractor = ag_fasta.FastaExtractor(fasta_path)
        self._encoder = one_hot_encoder.DNAOneHotEncoder()
        self._resolution = variant_scoring.get_resolution(dna_output.OutputType.RNA_SEQ)
        self._create_mask = functools.partial(
            cm.create_center_mask,
            width=center_width,
            resolution=self._resolution,
        )
        self._cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_size = cache_size
        self._genome_io = genome_io

    def _variant_key(self, chrom: str, pos: int, ref: str, alt: str) -> tuple:
        return (normalize_chrom(chrom), int(pos), str(ref).upper(), str(alt).upper())

    def get(self, chrom: str, pos: int, ref: str, alt: str):
        key = self._variant_key(chrom, pos, ref, alt)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        vrow = VariantRow(chrom=key[0], pos=key[1], ref=key[2], alt=key[3])
        variant = make_variant(vrow)
        interval = make_window_interval(variant, self.window)
        ref_seq, alt_seq = self._genome_io.extract_variant_sequences(interval, variant, self._extractor)
        ref_oh = self._encoder.encode(ref_seq).astype(np.float32)
        alt_oh = self._encoder.encode(alt_seq).astype(np.float32)
        mask = self._create_mask(interval, variant).astype(np.bool_)

        self._cache[key] = (ref_oh, alt_oh, mask)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return ref_oh, alt_oh, mask

    def stack_batch(self, chroms, positions, refs, alts):
        refs_list, alts_list, masks_list = [], [], []
        for c, p, r, a in zip(chroms, positions, refs, alts, strict=True):
            ref_oh, alt_oh, mask = self.get(c, p, r, a)
            refs_list.append(ref_oh)
            alts_list.append(alt_oh)
            masks_list.append(mask)
        return (
            np.stack(refs_list, axis=0),
            np.stack(alts_list, axis=0),
            np.stack(masks_list, axis=0),
        )
