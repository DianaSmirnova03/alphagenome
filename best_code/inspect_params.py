"""Печатает имена параметров AlphaGenome, совпадающих с паттерном.

Нужно перед `finetune_backbone.py`, чтобы узнать точное имя последнего слоя
RNA_SEQ-головы (структура параметров может отличаться между версиями/фолдами
модели, поэтому имя не хардкодится).

Запуск:
    python inspect_params.py --pattern rna_seq
"""

from __future__ import annotations

import argparse

from common import load_alphagenome_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="rna_seq")
    parser.add_argument("--model-version", default="all_folds")
    args = parser.parse_args()

    model = load_alphagenome_model(model_version=args.model_version)
    params = model._params  # noqa: SLF001 — см. common.py

    print(f"Модули, чьё имя содержит '{args.pattern}':\n")
    for module_name, module_params in params.items():
        if args.pattern.lower() in module_name.lower():
            for param_name, array in module_params.items():
                print(f"  {module_name}/{param_name}  shape={tuple(array.shape)}")


if __name__ == "__main__":
    main()
