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
from pathlib import Path

from src.dataset.prepare import prepare_dataset
from src.utils.utils import PROJECT_ROOT, get_logger, load_config

MODEL_CHOICES = ("yolo", "faster_rcnn", "ssd", "efficientdet", "detr")


def _train_yolo(config: dict, logger) -> None:
    """Запустить обучение baseline YOLOv8 на подготовленных данных.

    Путь к ``data.yaml`` берётся из ``paths.processed_data`` конфигурации
    (не хардкодится). Число эпох — baseline-значение из
    :mod:`src.training.train` (см. ``BASELINE_EPOCHS``), не финальные 50.
    """
    from src.models.yolo import build_model
    from src.training.train import train

    processed = Path(config.get("paths", {}).get("processed_data", "data/processed"))
    if not processed.is_absolute():
        processed = PROJECT_ROOT / processed
    data_yaml = processed / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Не найден {data_yaml}. Сначала подготовьте данные: "
            f"python main.py --prepare-data"
        )

    num_classes = len(config.get("dataset", {}).get("categories", []))
    model = build_model(num_classes, config=config, data_yaml=str(data_yaml))
    metrics = train(model, str(data_yaml), config, logger=logger)
    logger.info(
        "Готово. test mAP@0.5=%.4f, mAP@0.5:0.95=%.4f, P=%.4f, R=%.4f, F1=%.4f",
        metrics["map50"], metrics["map50_95"],
        metrics["precision"], metrics["recall"], metrics["f1"],
    )


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
        if args.model == "yolo":
            _train_yolo(config, logger)
        else:
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
