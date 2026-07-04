# Модель Faster R-CNN (двухэтапный детектор).
#
# Берём готовую fasterrcnn_resnet50_fpn из torchvision, обученную на COCO, и
# дообучаем под наши классы: меняем только последний классификатор
# (box_predictor). Это обычная сеть на PyTorch: при обучении она возвращает
# лоссы, а при проверке - предсказания. Обучается тем же общим кодом, что и SSD.

from __future__ import annotations


def build_model(num_classes: int, config: dict | None = None):
    """Создать Faster R-CNN (ResNet50-FPN) под наше число классов.

    num_classes - полное число классов вместе с фоном: 11 классов одежды + 1
    фон = 12. В torchvision класс 0 - это фон, поэтому метки объектов сдвинуты
    на +1 (см. coco_dataset.py).
    config - настройки (нужен просто для единого вида функций).

    Возвращает модель Faster R-CNN с новым классификатором.
    """
    try:
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as error:  # библиотека может быть не установлена
        raise ImportError(
            "Для Faster R-CNN требуется torchvision. "
            "Установите зависимости: pip install -r requirements.txt"
        ) from error

    # Размер картинки берём из конфига. Уменьшаем стандартный размер, чтобы
    # обучение шло быстрее, и делаем его как у YOLOv8 (640) для честного сравнения.
    image_size = (config or {}).get("training", {}).get("image_size", 800)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=int(image_size),
        max_size=int(image_size * 1.66),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
