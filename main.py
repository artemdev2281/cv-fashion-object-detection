"""Точка входа в проект Fashion Object Detection.

Поддерживаемые режимы:

* ``--prepare-data`` — загрузка и предобработка датасета Fashionpedia;
* ``--model <name>`` — обучение выбранной модели (``yolo``, ``faster_rcnn``,
  ``ssd``, ``efficientdet``, ``detr``) на подготовленных данных.

Типовой сценарий:

    python main.py --prepare-data
    python main.py --model yolo

Все параметры берутся из ``configs/default.yaml`` (пути, seed, число классов,
``dataset.subset_size`` — объём обучающей подвыборки).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.prepare import prepare_dataset
from src.utils.utils import PROJECT_ROOT, get_logger, load_config

MODEL_CHOICES = ("yolo", "faster_rcnn", "ssd", "efficientdet", "detr")


def _resolve_processed(config: dict) -> Path:
    """Абсолютный путь к каталогу подготовленных данных из конфигурации."""
    processed = Path(config.get("paths", {}).get("processed_data", "data/processed"))
    if not processed.is_absolute():
        processed = PROJECT_ROOT / processed
    return processed


def _report_metrics(model_name: str, metrics: dict, logger) -> None:
    """Единообразный вывод итоговых метрик на test."""
    logger.info(
        "Готово [%s]. test mAP@0.5=%.4f, mAP@0.5:0.95=%.4f, P=%.4f, R=%.4f, F1=%.4f",
        model_name, metrics["map50"], metrics["map50_95"],
        metrics["precision"], metrics["recall"], metrics["f1"],
    )


def _coco_inputs(config: dict) -> tuple[Path, Path, Path, list[str], int | None]:
    """Собрать входы для torchvision/effdet/detr обучения из подготовленных данных.

    Возвращает ``(train_ann, test_ann, data_root, class_names, subset_size)``.
    ``data_root`` — корень ``data/processed`` (в COCO JSON ``file_name`` задан
    относительно него). ``subset_size`` — обучающая подвыборка из
    ``training.subset_size`` (``null`` = полный train); это отдельный параметр от
    ``dataset.subset_size``, который ограничивает объём на этапе подготовки.
    Итоговая оценка — всегда на полном test.
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
    """Обучить YOLOv8 на подготовленных данных (путь к ``data.yaml`` из конфига)."""
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
    """Обучить Faster R-CNN или SSD общим torchvision-раннером.

    ``num_classes = 11 + 1`` (класс 0 — фон, метки сдвинуты на +1); оптимизатор,
    lr и прочее — дефолты раннера (SGD lr 0.005), совпадающие с использованными
    в экспериментах.
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
    """Обучить EfficientDet-D0 (пакет ``effdet``) на подготовленных данных."""
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
    """Обучить DETR-R50 (HuggingFace ``transformers``) на подготовленных данных."""
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


#: Диспетчер моделей: имя -> функция запуска обучения.
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
