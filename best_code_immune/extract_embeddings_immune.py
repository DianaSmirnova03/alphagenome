"""Извлечение признаков для иммунных ASE-данных из СЫРЫХ внутренних
эмбеддингов AlphaGenome (`embeddings_128bp`, выход `TransformerTower`,
3072-дим на каждые 128 п.н.), а не из предсказанных RNA_SEQ-треков
(`extract_features_immune.py` / `CenterMaskScorer`).

Мотивация: в курсовой (раздел 5.1, "Эмбеддинги alphagenome") RF, обученный
на разности сырых внутренних представлений ref/alt (усреднённой по длине),
дал AUC 0.763 на direction (over/under) для этих же данных AIDA — заметно
выше, чем то, что удалось получить на признаках из предсказанных RNA_SEQ
треков (0.669–0.766). Точный скрипт с этим результатом не найден в
репозитории (не сохранился отдельным файлом), поэтому здесь он
воспроизводится с нуля, максимально близко к описанию в курсовой:
"усредняли по длине разность выхода модели для alt и ref последовательности".

Отличия от `extract_features_immune.py`:
- Используется `common.build_embeddings_apply_fn` (дифференцируемый форвард,
  напрямую из `alphagenome_research`) вместо `model.score_variant`.
- Признак — это `mean_over_length(embeddings_alt) - mean_over_length(embeddings_ref)`,
  3072-дим вектор (а не 667 RNA_SEQ треков и не 2001-п.н. center-mask).
- Батчинг по несколько вариантов одновременно (--batch-size) для скорости.
- Та же дедупликация по уникальному варианту (comb_es/fdr/cell_type НЕ influence
  признак — он зависит только от последовательности).

Запуск (на одном свободном GPU-ядре):
    CUDA_VISIBLE_DEVICES=0 python extract_embeddings_immune.py \
        --h5 ../chr_data/chr_train.h5 --split train_all \
        --fasta AG/hg38.fa --out-dir features_immune_emb \
        --vocab-path features_immune/cell_type_vocab.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    VariantRow,
    build_embeddings_apply_fn,
    load_alphagenome_model,
    make_variant,
    make_window_interval,
    normalize_chrom,
    set_seed,
)


def _load_h5(h5_path: str) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as f:
        chrom_raw = f["chromosome"][()]
        df = pd.DataFrame(
            {
                "chrom": [normalize_chrom(c) for c in chrom_raw],
                "pos": f["position_start"][()].astype(int) + 1,  # 0-based BED -> 1-based
                "ref": f["ref"][()].astype(str),
                "alt": f["alt"][()].astype(str),
                "cell_type": f["cell_type"][()].astype(str),
                "comb_es": f["comb_es"][()].astype(np.float32),
                "fdr_comb_pval": f["fdr_comb_pval"][()].astype(np.float32),
                "variant_id": f["variant_id"][()].astype(str),
            }
        )
    return df


def extract_embeddings_immune(
    h5_path: str,
    split: str,
    fasta_path: str,
    out_dir: str,
    window: int = 131072,
    model_version: str = "all_folds",
    limit: Optional[int] = None,
    vocab_path: Optional[str] = None,
    batch_size: int = 4,
) -> None:
    from alphagenome.models.dna_model import Organism
    from alphagenome_research.io import fasta as ag_fasta
    from alphagenome_research.io import genome as genome_io
    from alphagenome_research.model import dna_model as ag_dna_model
    from alphagenome_research.model import one_hot_encoder

    set_seed(0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_h5(h5_path)
    print(f"Загружено {len(df)} строк из {h5_path} ({df['chrom'].nunique()} хромосом)")

    if vocab_path is not None and Path(vocab_path).exists():
        cell_type_vocab = json.loads(Path(vocab_path).read_text())
        print(f"Используется существующий словарь cell_type из {vocab_path} ({len(cell_type_vocab)} типов)")
    else:
        cell_type_vocab = sorted(df["cell_type"].unique().tolist())
        if vocab_path is not None:
            Path(vocab_path).write_text(json.dumps(cell_type_vocab, ensure_ascii=False, indent=2))
    unknown_idx = len(cell_type_vocab)

    if limit is not None:
        key_cols = ["chrom", "pos", "ref", "alt"]
        keep_keys = df.drop_duplicates(key_cols).head(limit)[key_cols]
        df = df.merge(keep_keys, on=key_cols, how="inner")

    key_cols = ["chrom", "pos", "ref", "alt"]
    unique_variants = df.drop_duplicates(key_cols)[key_cols].reset_index(drop=True)
    print(f"Уникальных вариантов: {len(unique_variants)} (из {len(df)} строк)")

    model = load_alphagenome_model(model_version=model_version, fasta_path=fasta_path)
    embeddings_apply_fn = build_embeddings_apply_fn(model)
    params, state = model._params, model._state  # noqa: SLF001 — см. common.py
    organism_index = ag_dna_model.convert_to_organism_index(Organism.HOMO_SAPIENS)

    extractor = ag_fasta.FastaExtractor(fasta_path)
    encoder = one_hot_encoder.DNAOneHotEncoder()

    @jax.jit
    def embed_batch(sequences, organism_idx_batch):
        # sequences: [B, S, 4] -> embeddings_128bp: [B, S//128, 3072] -> mean по позиции
        emb = embeddings_apply_fn(params, state, sequences, organism_idx_batch)
        return jnp.mean(emb.astype(jnp.float32), axis=1)

    lfc_cache: dict[tuple, np.ndarray] = {}
    n_error = 0

    rows = list(unique_variants.itertuples(index=False))
    for start in tqdm(range(0, len(rows), batch_size), desc=f"extract_emb[{split}]"):
        chunk = rows[start : start + batch_size]
        ref_batch, alt_batch, keys = [], [], []
        for r in chunk:
            key = (r.chrom, int(r.pos), r.ref, r.alt)
            try:
                vrow = VariantRow(chrom=r.chrom, pos=int(r.pos), ref=r.ref, alt=r.alt)
                variant = make_variant(vrow)
                interval = make_window_interval(variant, window)
                ref_seq, alt_seq = genome_io.extract_variant_sequences(interval, variant, extractor)
                ref_batch.append(encoder.encode(ref_seq).astype(np.float32))
                alt_batch.append(encoder.encode(alt_seq).astype(np.float32))
                keys.append(key)
            except Exception as exc:  # noqa: BLE001
                n_error += 1
                tqdm.write(f"[warn] пропущен вариант {r.chrom}:{r.pos} ({exc})")
        if not keys:
            continue

        ref_arr = jnp.asarray(np.stack(ref_batch, axis=0))
        alt_arr = jnp.asarray(np.stack(alt_batch, axis=0))
        organism_idx_batch = jnp.full((len(keys),), organism_index, dtype=jnp.int32)

        ref_mean = np.asarray(embed_batch(ref_arr, organism_idx_batch))
        alt_mean = np.asarray(embed_batch(alt_arr, organism_idx_batch))
        diff = alt_mean - ref_mean  # [B, 3072]

        for i, key in enumerate(keys):
            lfc_cache[key] = diff[i].astype(np.float32)

    print(f"Готово: {len(lfc_cache)} уникальных вариантов успешно, {n_error} с ошибкой.")
    if not lfc_cache:
        raise RuntimeError("Ни одного варианта не удалось обработать — проверьте --fasta.")

    X, cell_type_idx, comb_es, fdr, chroms, poss = [], [], [], [], [], []
    kept_variant_id = []
    n_skipped = 0
    for _, row in df.iterrows():
        key = (row["chrom"], int(row["pos"]), row["ref"], row["alt"])
        feat = lfc_cache.get(key)
        if feat is None:
            n_skipped += 1
            continue
        X.append(feat)
        cell_type_idx.append(cell_type_vocab.index(row["cell_type"]) if row["cell_type"] in cell_type_vocab else unknown_idx)
        comb_es.append(row["comb_es"])
        fdr.append(row["fdr_comb_pval"])
        chroms.append(row["chrom"])
        poss.append(row["pos"])
        kept_variant_id.append(row["variant_id"])

    X = np.stack(X, axis=0)
    cell_type_idx = np.asarray(cell_type_idx, dtype=np.int64)
    comb_es = np.asarray(comb_es, dtype=np.float32)
    fdr = np.asarray(fdr, dtype=np.float32)
    is_sig = (fdr < 0.05).astype(np.float32)

    np.savez(
        out_dir / f"{split}_features.npz",
        X=X,
        cell_type_idx=cell_type_idx,
        comb_es=comb_es,
        fdr_comb_pval=fdr,
        is_sig=is_sig,
        chrom=np.asarray(chroms),
        pos=np.asarray(poss, dtype=np.int64),
        variant_id=np.asarray(kept_variant_id),
    )
    print(
        f"Сохранено: {out_dir / f'{split}_features.npz'} (X shape={X.shape}, строк пропущено: {n_skipped})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=int, default=131072)
    parser.add_argument("--model-version", default="all_folds")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    extract_embeddings_immune(
        h5_path=args.h5,
        split=args.split,
        fasta_path=args.fasta,
        out_dir=args.out_dir,
        window=args.window,
        model_version=args.model_version,
        limit=args.limit,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
