"""Модель SSD (Single Shot MultiBox Detector, одноэтапный детектор).

Fine-tuning предобученной на COCO ``ssd300_vgg16`` из
``torchvision.models.detection``. У SSD голова устроена иначе, чем у
Faster R-CNN: заменяется не ``FastRCNNPredictor``, а classification-часть
головы (``SSDClassificationHead``), при этом число якорей на позицию и число
входных каналов сохраняются от исходной сети. Модель обучается тем же общим
циклом :func:`src.training.train.train_torchvision_detector`.
"""

from __future__ import annotations


def build_model(num_classes: int, config: dict | None = None):
    """Создать SSD300 (VGG16) под ``num_classes``.

    Параметры
    ---------
    num_classes:
        **Полное** число классов, включая фон: 11 категорий + 1 фон = 12
        (класс 0 — фон, метки объектов сдвинуты на +1, см.
        :mod:`src.dataset.coco_dataset`).
    config:
        Конфигурация эксперимента (для совместимости интерфейса).

    Возвращает
    ----------
    ``torchvision.models.detection.SSD`` с новой classification-головой.
    """
    try:
        import torchvision
        from torchvision.models.detection import _utils as det_utils
        from torchvision.models.detection.ssd import SSDClassificationHead
    except ImportError as error:  # pragma: no cover - зависит от среды
        raise ImportError(
            "Для SSD требуется torchvision. "
            "Установите зависимости: pip install -r requirements.txt"
        ) from error

    model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
    # Число входных каналов на каждой карте признаков и число якорей на позицию
    # берём от исходной сети, чтобы заменить только число классов.
    in_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes
    )
    return model
