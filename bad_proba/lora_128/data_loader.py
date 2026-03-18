import numpy as np
import pandas as pd
import jax
import pyfaidx
import jax.numpy as jnp
from alphagenome_research.model import one_hot_encoder

WINDOW = 20480
HALF = WINDOW // 2
COMPLEMENT = str.maketrans('ACGT', 'TGCA')

class PromoterDataset:
    def __init__(self, csv_path, fasta_path, batch_size=8, shuffle=True, limit=None):
        self.df = pd.read_csv(csv_path)
        if limit:
            self.df = self.df.iloc[:limit]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)
        self._rng_key = jax.random.PRNGKey(42)

    def __len__(self):
        return len(self.df)

    def _get_sequences(self, row):
        chrom, pos, ref, alt, strand = row['chrom'], row['pos'], row['ref'], row['alt'], row['strand']
        start = max(0, pos - HALF)
        end = pos + HALF
        seq_phys = self.fasta[chrom][start:end].seq.upper()
        if len(seq_phys) < WINDOW:
            seq_phys = seq_phys.ljust(WINDOW, 'N')
        offset = pos - start
        seq_alt_phys = seq_phys[:offset] + alt + seq_phys[offset+1:]

        if strand == -1:
            seq_ref = seq_phys.translate(COMPLEMENT)[::-1]
            seq_alt = seq_alt_phys.translate(COMPLEMENT)[::-1]
        else:
            seq_ref = seq_phys
            seq_alt = seq_alt_phys

        ref_oh = self.encoder.encode(seq_ref).astype(np.float32)
        alt_oh = self.encoder.encode(seq_alt).astype(np.float32)
        return ref_oh, alt_oh

    def generate(self):
        indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start+self.batch_size]
            batch_ref = []
            batch_alt = []
            targets = []
            for idx in batch_idx:
                row = self.df.iloc[idx]
                ref_oh, alt_oh = self._get_sequences(row)
                batch_ref.append(ref_oh)
                batch_alt.append(alt_oh)
                targets.append(row['z'])
            batch_ref = np.stack(batch_ref)
            batch_alt = np.stack(batch_alt)
            targets = np.array(targets, dtype=np.float32)
            yield batch_ref, batch_alt, targets
