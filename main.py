"""Точка входа в проект Fashion Object Detection.

Поддерживаемые режимы:

* ``--prepare-data`` — загрузка и предобработка датасета Fashionpedia;
* ``--model <name>`` — обучение выбранной модели (будет реализовано на этапе
  экспериментов).

Пример запуска в Kaggle::

    !python main.py --prepare-data
"""

from __future__ import annotations

import argparse

from src.dataset.prepare import prepare_dataset
from src.utils.utils import get_logger, load_config

MODEL_CHOICES = ("yolo", "faster_rcnn", "ssd", "efficientdet", "detr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fashion Object Detection — подготовка данных и обучение моделей."
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="загрузить и предобработать датасет Fashionpedia",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        help="модель для обучения (будет реализовано на этапе экспериментов)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="путь к YAML-конфигурации (по умолчанию configs/default.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    logger = get_logger()

    if args.prepare_data:
        prepare_dataset(config, logger)
    elif args.model:
        logger.info(
            "Обучение модели '%s' пока не реализовано "
            "(будет добавлено на этапе экспериментов).",
            args.model,
        )
    else:
        logger.info(
            "Не указан режим. Используйте --prepare-data или --model <name>."
        )


if __name__ == "__main__":
    main()
