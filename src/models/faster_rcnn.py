"""Модель Faster R-CNN (двухэтапный детектор с сетью предложения областей).

Fine-tuning предобученной на COCO ``fasterrcnn_resnet50_fpn`` из
``torchvision.models.detection``: заменяется финальный классификатор
(``box_predictor``) под число классов проекта. Модель — обычный
``torch.nn.Module`` с собственным протоколом detection (в режиме ``train`` при
переданных таргетах возвращает словарь лоссов, в ``eval`` — предсказания),
поэтому обучается общим циклом
:func:`src.training.train.train_torchvision_detector`.
"""

from __future__ import annotations


def build_model(num_classes: int, config: dict | None = None):
    """Создать Faster R-CNN (ResNet50-FPN) под ``num_classes``.

    Параметры
    ---------
    num_classes:
        **Полное** число классов, включая фон: 11 категорий одежды + 1 фон = 12.
        В torchvision detection класс 0 зарезервирован под фон; метки объектов
        поэтому подаются сдвинутыми на +1 (см. :mod:`src.dataset.coco_dataset`).
    config:
        Конфигурация эксперимента (для совместимости интерфейса; гиперпараметры
        обучения передаются в цикл обучения, а не в конструктор).

    Возвращает
    ----------
    ``torchvision.models.detection.FasterRCNN`` с новым ``box_predictor``.
    """
    try:
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as error:  # pragma: no cover - зависит от среды
        raise ImportError(
            "Для Faster R-CNN требуется torchvision. "
            "Установите зависимости: pip install -r requirements.txt"
        ) from error

    # Разрешение входа берём из конфига (training.image_size). Понижение с
    # дефолтных 800/1333 ускоряет обучение на Tesla T4 в разы; согласуем с
    # image_size YOLOv8 (640) для сопоставимости масштаба.
    image_size = (config or {}).get("training", {}).get("image_size", 800)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=int(image_size),
        max_size=int(image_size * 1.66),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
