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

from src.dataset.dataset import prepare_data
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


def run_prepare_data(config: dict, logger) -> None:
    dataset_cfg = config.get("dataset", {})
    splits_cfg = dataset_cfg.get("splits", {})
    training_cfg = config.get("training", {})

    logger.info("Загрузка и предобработка датасета Fashionpedia...")
    dataset = prepare_data(
        categories=dataset_cfg.get("categories"),
        test_size=splits_cfg.get("test_size", 0.1),
        subset_size=dataset_cfg.get("subset_size"),
        streaming=dataset_cfg.get("streaming", False),
        seed=training_cfg.get("seed", 42),
    )

    logger.info("Подготовка данных завершена. Размеры выборок:")
    for split, data in dataset.items():
        try:
            size = len(data)
        except TypeError:
            size = "потоковый режим (размер неизвестен)"
        logger.info("  %s: %s", split, size)


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    logger = get_logger()

    if args.prepare_data:
        run_prepare_data(config, logger)
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
