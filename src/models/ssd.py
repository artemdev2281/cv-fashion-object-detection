# Модель SSD (Single Shot MultiBox Detector).
#
# Берём готовую ssd300_vgg16 из torchvision, обученную на COCO, и дообучаем под
# наши классы. У SSD голова устроена не так, как у Faster R-CNN: тут меняем
# именно ту часть, которая отвечает за классы (SSDClassificationHead), а
# остальное оставляем как есть. Обучается тем же общим кодом, что и Faster R-CNN.

from __future__ import annotations


def build_model(num_classes: int, config: dict | None = None):
    """Создать SSD300 (VGG16) под наше число классов.

    num_classes - полное число классов вместе с фоном: 11 классов + 1 фон = 12
    (класс 0 - фон, метки объектов сдвинуты на +1, см. coco_dataset.py).
    config - настройки (нужен просто для единого вида функций).

    Возвращает модель SSD с новой головой под наши классы.
    """
    try:
        import torchvision
        from torchvision.models.detection import _utils as det_utils
        from torchvision.models.detection.ssd import SSDClassificationHead
    except ImportError as error:  # библиотека может быть не установлена
        raise ImportError(
            "Для SSD требуется torchvision. "
            "Установите зависимости: pip install -r requirements.txt"
        ) from error

    model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
    # Размеры для новой головы берём у исходной сети, чтобы поменять только
    # число классов, а всё остальное оставить прежним.
    in_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes
    )
    return model
