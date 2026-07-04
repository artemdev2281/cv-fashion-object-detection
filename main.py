# Главный файл проекта. Отсюда запускаем всё.
#
# Что умеет:
#   python main.py --prepare-data     - скачать и подготовить датасет
#   python main.py --model yolo       - обучить модель (yolo, faster_rcnn,
#                                       ssd, efficientdet, detr)
#
# Настройки лежат в configs/default.yaml.

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.prepare import prepare_dataset
from src.utils.utils import PROJECT_ROOT, get_logger, load_config

MODEL_CHOICES = ("yolo", "faster_rcnn", "ssd", "efficientdet", "detr")


def _resolve_processed(config: dict) -> Path:
    """Получить путь к папке с подготовленными данными."""
    processed = Path(config.get("paths", {}).get("processed_data", "data/processed"))
    if not processed.is_absolute():
        processed = PROJECT_ROOT / processed
    return processed


def _report_metrics(model_name: str, metrics: dict, logger) -> None:
    """Вывести итоговые метрики модели на тесте."""
    logger.info(
        "Готово [%s]. test mAP@0.5=%.4f, mAP@0.5:0.95=%.4f, P=%.4f, R=%.4f, F1=%.4f",
        model_name, metrics["map50"], metrics["map50_95"],
        metrics["precision"], metrics["recall"], metrics["f1"],
    )


def _coco_inputs(config: dict) -> tuple[Path, Path, Path, list[str], int | None]:
    """Собрать пути к данным для обучения faster_rcnn/ssd/efficientdet/detr.

    Возвращает пути к train.json и test.json, корень с картинками, список
    классов и размер подвыборки для обучения (training.subset_size). Тест
    всегда считаем на всех картинках.
    """
    processed = _resolve_processed(config)
    train_ann = processed / "coco" / "train.json"
    test_ann = processed / "coco" / "test.json"
    if not train_ann.exists() or not test_ann.exists():
        raise FileNotFoundError(
            f"Не найдены COCO-аннотации в {processed / 'coco'}. Сначала подготовьте "
            f"данные: python main.py --prepare-data"
        )
    class_names = list(config.get("dataset", {}).get("categories", []))
    subset_size = config.get("training", {}).get("subset_size")
    return train_ann, test_ann, processed, class_names, subset_size


def _train_yolo(config: dict, logger) -> None:
    """Обучить YOLOv8 (ему нужен файл data.yaml)."""
    from src.models.yolo import build_model
    from src.training.train import train

    data_yaml = _resolve_processed(config) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Не найден {data_yaml}. Сначала подготовьте данные: "
            f"python main.py --prepare-data"
        )

    if config.get("training", {}).get("subset_size"):
        logger.warning(
            "training.subset_size задан, но к YOLOv8 здесь не применяется: "
            "Ultralytics обучается на всём data.yaml. Для обучения YOLO на "
            "подвыборке используйте subset-data.yaml из notebooks/train_yolov8.ipynb."
        )
    num_classes = len(config.get("dataset", {}).get("categories", []))
    model = build_model(num_classes, config=config, data_yaml=str(data_yaml))
    metrics = train(model, str(data_yaml), config, logger=logger)
    _report_metrics("yolo", metrics, logger)


def _train_torchvision(config: dict, logger, *, model_name: str, batch_size: int) -> None:
    """Обучить Faster R-CNN или SSD (у них общий код обучения).

    num_classes = 11 классов + 1 фон = 12 (в torchvision класс 0 - это фон,
    поэтому метки сдвинуты на +1). Оптимизатор и learning rate берутся по
    умолчанию (SGD, lr 0.005).
    """
    from src.training.train import run_torchvision_training

    if model_name == "faster_rcnn":
        from src.models.faster_rcnn import build_model
    else:
        from src.models.ssd import build_model

    train_ann, test_ann, data_root, class_names, subset_size = _coco_inputs(config)
    logger.info(
        "%s: обучение на подвыборке train=%s (eval — полный test)",
        model_name, subset_size if subset_size else "полный",
    )
    metrics = run_torchvision_training(
        build_model, len(class_names) + 1, train_ann, test_ann, data_root, config,
        model_name=model_name, class_names=class_names, batch_size=batch_size,
        label_offset=1, subset_size=subset_size, logger=logger,
    )
    _report_metrics(model_name, metrics, logger)


def _train_efficientdet(config: dict, logger) -> None:
    """Обучить EfficientDet-D0 (через библиотеку effdet)."""
    from src.training.train import run_efficientdet_training

    train_ann, test_ann, data_root, class_names, subset_size = _coco_inputs(config)
    logger.info(
        "efficientdet: обучение на подвыборке train=%s (eval — полный test)",
        subset_size if subset_size else "полный",
    )
    metrics = run_efficientdet_training(
        len(class_names), train_ann, test_ann, data_root, config,
        class_names=class_names, batch_size=4, image_size=512,
        subset_size=subset_size, logger=logger,
    )
    _report_metrics("efficientdet", metrics, logger)


def _train_detr(config: dict, logger) -> None:
    """Обучить DETR-R50 (через библиотеку transformers)."""
    from src.training.train import run_detr_training

    train_ann, test_ann, data_root, class_names, subset_size = _coco_inputs(config)
    logger.info(
        "detr: обучение на подвыборке train=%s (eval — полный test)",
        subset_size if subset_size else "полный",
    )
    metrics = run_detr_training(
        len(class_names), train_ann, test_ann, data_root, config,
        class_names=class_names, batch_size=2,
        subset_size=subset_size, logger=logger,
    )
    _report_metrics("detr", metrics, logger)


# По имени модели выбираем нужную функцию обучения.
_TRAINERS = {
    "yolo": _train_yolo,
    "faster_rcnn": lambda c, l: _train_torchvision(c, l, model_name="faster_rcnn", batch_size=4),
    "ssd": lambda c, l: _train_torchvision(c, l, model_name="ssd", batch_size=8),
    "efficientdet": _train_efficientdet,
    "detr": _train_detr,
}


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
        help="модель для обучения: yolo | faster_rcnn | ssd | efficientdet | detr",
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
        _TRAINERS[args.model](config, logger)
    else:
        logger.info(
            "Не указан режим. Используйте --prepare-data или --model <name> "
            "(%s).", " | ".join(MODEL_CHOICES),
        )


if __name__ == "__main__":
    main()
