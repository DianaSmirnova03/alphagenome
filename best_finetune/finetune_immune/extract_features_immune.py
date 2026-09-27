"""Извлечение признаков для иммунных ASE-данных (`chr_train.h5` / `chr_test.h5`).

Отличие от `extract_features.py` (SNP -> экспрессия гена по z-score):
здесь целевая переменная — `comb_es`, комбинированный effect size
allele-specific expression (по сути мера того, во сколько раз чаще
встречаются риды с alt-аллелем, чем с ref, в РНК-seq иммунных клеток),
измеренная **по типу иммунной клетки** (`cell_type`, 37 категорий: T/B/NK/
моноциты/DC и их подтипы). У данных нет привязки к конкретному гену
(`gene_id` в файле отсутствует) — поэтому вместо официального
`GeneMaskLFCScorer` (агрегация по маске экзонов гена) используется другой
официальный скорер AlphaGenome — `CenterMaskScorer`: та же идея
(log2(alt/ref) на предсказанных RNA_SEQ треках), но агрегация — по
фиксированному окну шириной `--center-width` п.н., отцентрированному прямо
на самом варианте (используется для CAGE/PROCAP/ATAC в официальных
рекомендованных скорерах AlphaGenome — здесь применяется к RNA_SEQ, чтобы
получить локальный, не размазанный по всему гену сигнал ровно в точке SNP).

Дедупликация: один и тот же вариант (chrom,pos,ref,alt) в HDF5 встречается
много раз — по разу на каждый cell_type, для которого была измерена ASE.
Признак модели (LFC по трекам) зависит только от последовательности,
поэтому считается **один раз на уникальный вариант** и копируется на все
строки с этим вариантом. Без дедупликации (как в старом
`AG/emb_jax_immune.py`) пришлось бы пересчитывать одну и ту же
последовательность в ~4 раза больше раз, чем нужно (81273 строки против
19843 уникальных вариантов на train).

Запуск:
    python extract_features_immune.py --h5 ../chr_data/chr_train.h5 --split train_all \
        --fasta AG/hg38.fa --out-dir features_immune --limit 200   # для быстрого теста
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    VariantRow,
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
                # HDF5 хранит 0-based BED-style координату (position_end ==
                # position_start + 1); проверено напрямую по hg38.fa:
                # fasta[chrom][position_start] == ref. genome.Variant.position
                # ожидает 1-based -> +1.
                "pos": f["position_start"][()].astype(int) + 1,
                "ref": f["ref"][()].astype(str),
                "alt": f["alt"][()].astype(str),
                "cell_type": f["cell_type"][()].astype(str),
                "comb_es": f["comb_es"][()].astype(np.float32),
                "fdr_comb_pval": f["fdr_comb_pval"][()].astype(np.float32),
                "variant_id": f["variant_id"][()].astype(str),
            }
        )
    return df


def extract_features_immune(
    h5_path: str,
    split: str,
    fasta_path: str,
    out_dir: str,
    window: int = 131072,
    center_width: int = 2001,
    model_version: str = "all_folds",
    limit: Optional[int] = None,
    vocab_path: Optional[str] = None,
) -> None:
    from alphagenome.models import dna_output, variant_scorers

    set_seed(0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_h5(h5_path)
    print(f"Загружено {len(df)} строк из {h5_path} ({df['chrom'].nunique()} хромосом)")

    # Словарь cell_type -> индекс. Строится один раз (обычно на train_all) и
    # переиспользуется для val/test, чтобы кодировка была одинаковой везде.
    if vocab_path is not None and Path(vocab_path).exists():
        cell_type_vocab = json.loads(Path(vocab_path).read_text())
        print(f"Используется существующий словарь cell_type из {vocab_path} ({len(cell_type_vocab)} типов)")
    else:
        cell_type_vocab = sorted(df["cell_type"].unique().tolist())
        if vocab_path is not None:
            Path(vocab_path).write_text(json.dumps(cell_type_vocab, ensure_ascii=False, indent=2))
            print(f"Сохранён словарь cell_type: {vocab_path} ({len(cell_type_vocab)} типов)")
    unknown_idx = len(cell_type_vocab)  # запасной индекс для неизвестных типов

    if limit is not None:
        # ограничиваем ЧИСЛО УНИКАЛЬНЫХ ВАРИАНТОВ, а не строк, чтобы быстрый
        # тест реально прогнал через модель --limit разных последовательностей
        key_cols = ["chrom", "pos", "ref", "alt"]
        keep_keys = df.drop_duplicates(key_cols).head(limit)[key_cols]
        df = df.merge(keep_keys, on=key_cols, how="inner")

    key_cols = ["chrom", "pos", "ref", "alt"]
    unique_variants = df.drop_duplicates(key_cols)[key_cols].reset_index(drop=True)
    print(f"Уникальных вариантов: {len(unique_variants)} (из {len(df)} строк, дедупликация по cell_type)")

    model = load_alphagenome_model(model_version=model_version, fasta_path=fasta_path)
    scorer = variant_scorers.CenterMaskScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
        width=center_width,
        aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
    )

    lfc_cache: dict[tuple, np.ndarray] = {}
    n_error = 0
    track_metadata_saved = False

    for _, row in tqdm(unique_variants.iterrows(), total=len(unique_variants), desc=f"extract_immune[{split}]"):
        vrow = VariantRow(chrom=row["chrom"], pos=int(row["pos"]), ref=row["ref"], alt=row["alt"])
        key = (row["chrom"], int(row["pos"]), row["ref"], row["alt"])
        try:
            variant = make_variant(vrow)
            interval = make_window_interval(variant, window)
            results = model.score_variant(interval, variant, variant_scorers=[scorer])
            ann = results[0]
            lfc_cache[key] = np.asarray(ann.X[0, :], dtype=np.float32)
            if not track_metadata_saved:
                ann.var.reset_index(drop=True).to_csv(out_dir / "track_metadata_immune.csv", index=False)
                track_metadata_saved = True
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            tqdm.write(f"[warn] пропущен вариант {vrow.chrom}:{vrow.pos} ({exc})")

    print(f"Готово: {len(lfc_cache)} уникальных вариантов успешно, {n_error} с ошибкой.")
    if not lfc_cache:
        raise RuntimeError("Ни одного варианта не удалось обработать — проверьте --fasta.")

    # Разворачиваем кэш обратно на все строки (по одной на каждый cell_type)
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
        f"Сохранено: {out_dir / f'{split}_features.npz'} (X shape={X.shape}, строк пропущено из-за ошибок LFC: {n_skipped})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=int, default=131072)
    parser.add_argument("--center-width", type=int, default=2001)
    parser.add_argument("--model-version", default="all_folds")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число УНИКАЛЬНЫХ вариантов (для теста)")
    parser.add_argument(
        "--vocab-path",
        default=None,
        help="Путь к JSON-словарю cell_type. Если не существует — создаётся из текущего файла.",
    )
    args = parser.parse_args()

    extract_features_immune(
        h5_path=args.h5,
        split=args.split,
        fasta_path=args.fasta,
        out_dir=args.out_dir,
        window=args.window,
        center_width=args.center_width,
        model_version=args.model_version,
        limit=args.limit,
        vocab_path=args.vocab_path,
    )


if __name__ == "__main__":
    main()
