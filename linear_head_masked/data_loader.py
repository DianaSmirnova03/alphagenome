import numpy as np
import pandas as pd
import pyfaidx
import pickle
from alphagenome_research.model import one_hot_encoder

WINDOW = 131_072
HALF = WINDOW // 2

class VariantDataset:
    def __init__(self, csv_path, fasta_path, exon_pkl_path, batch_size=4, shuffle=True):
        self.df = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)

        with open(exon_pkl_path, "rb") as f:
            self.gene_exons = pickle.load(f)

        self.items = self._prepare_items()

    def _prepare_items(self):
        items = []
        for _, row in self.df.iterrows():
            chrom = row["chrom"]
            pos = int(row["pos"])
            ref = row["ref"]
            alt = row["alt"]
            strand = int(row["strand"])
            gene = row["gene"]
            z = float(row["z"])

            items.append({
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "strand": strand,
                "gene": gene,
                "z": z
            })
        return items

    def _get_gene_mask(self, gene, chrom, pos, strand):
        exons = self.gene_exons.get(gene, [])
        mask = np.zeros(WINDOW, dtype=np.float32)
        window_start = pos - HALF
        window_end = pos + HALF

        for (echrom, estart, eend, estrand) in exons:
            if echrom != chrom or estrand != strand:
                continue
            overlap_start = max(estart, window_start)
            overlap_end = min(eend, window_end)
            if overlap_end <= overlap_start:
                continue
            idx_start = overlap_start - window_start
            idx_end = overlap_end - window_start
            mask[idx_start:idx_end] = 1.0
        return mask

    def _get_seq(self, chrom, center, alt_base=None, strand=1):
        start = max(0, center - HALF)
        end = center + HALF
        seq = self.fasta[chrom][start:end].seq.upper()
        if len(seq) < WINDOW:
            seq = seq.ljust(WINDOW, 'N')
        if alt_base is not None:
            offset = center - start
            seq = seq[:offset] + alt_base + seq[offset+1:]
        if strand == -1:
            seq = seq.translate(str.maketrans('ACGT','TGCA'))[::-1]
        return seq

    def _encode(self, seq):
        return self.encoder.encode(seq).astype(np.float32)

    def generate(self):
        indices = np.arange(len(self.items))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx+self.batch_size]
            batch_ref = []
            batch_alt = []
            batch_mask = []
            batch_targets = []

            for idx in batch_indices:
                item = self.items[idx]
                pos = item["pos"]
                chrom = item["chrom"]
                strand = item["strand"]

                ref_seq = self._get_seq(chrom, pos, strand=strand)
                alt_seq = self._get_seq(chrom, pos, item["alt"], strand=strand)

                ref_oh = self._encode(ref_seq)
                alt_oh = self._encode(alt_seq)
                mask = self._get_gene_mask(item["gene"], chrom, pos, strand)
                target = item["z"]

                batch_ref.append(ref_oh)
                batch_alt.append(alt_oh)
                batch_mask.append(mask)
                batch_targets.append(target)

            yield {
                "ref_seq": np.stack(batch_ref),
                "alt_seq": np.stack(batch_alt),
                "gene_mask": np.stack(batch_mask),
                "targets": np.array(batch_targets, dtype=np.float32)
            }

    def __len__(self):
        return (len(self.items) + self.batch_size - 1) // self.batch_size
