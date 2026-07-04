# Обработка картинок перед обучением: изменение размера, нормализация и
# аугментации (случайные изменения картинки, чтобы модель лучше обучалась).
#
# Важно, что модели работают по-разному:
# - YOLOv8 всё это делает сам внутри (через настройки Ultralytics), поэтому
#   картинки на диске лежат обычные, а параметры аугментаций для него собирает
#   функция yolo_augmentation_args.
# - остальные модели (Faster R-CNN, SSD, EfficientDet, DETR) получают
#   преобразования через build_transforms.
#
# Аугментации применяем только к train. Для val и test делаем только resize
# и нормализацию, чтобы честно оценить качество.

from __future__ import annotations

from typing import Sequence

# Стандартные числа ImageNet для нормализации пикселей.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def yolo_augmentation_args(config: dict) -> dict:
    """Собрать настройки аугментаций для YOLOv8 из конфига.

    Берёт раздел augmentation из конфига и превращает его в словарь настроек,
    который понимает Ultralytics. Сами аугментации YOLOv8 делает внутри себя.
    """
    augmentation = config.get("augmentation", {})
    return {
        "fliplr": augmentation.get("horizontal_flip", 0.0),
        "hsv_s": augmentation.get("color_jitter", 0.0),
        "hsv_v": augmentation.get("color_jitter", 0.0),
        "mosaic": augmentation.get("mosaic", 0.0),
        # В Ultralytics обрезка (crop) включается через scale.
        "scale": 0.5 if augmentation.get("random_crop", False) else 0.0,
    }


def build_transforms(config: dict, split: str):
    """Собрать преобразования картинок для всех моделей, кроме YOLO.

    Используем torchvision.transforms.v2 - он умеет менять картинку вместе с
    рамками. Аугментации добавляем только для train.

    config - настройки, split - какая часть данных (train / val / test).
    Возвращает готовый набор преобразований.
    """
    try:
        import torch
        from torchvision.transforms import v2
    except ImportError as error:  # библиотека может быть не установлена
        raise ImportError(
            "Для build_transforms требуется torchvision (>=0.16) с API transforms.v2. "
            "Установите зависимости из requirements.txt."
        ) from error

    image_size = config.get("training", {}).get("image_size", 640)
    augmentation = config.get("augmentation", {})

    transforms: list = [v2.Resize((image_size, image_size))]

    if split == "train":
        flip_prob = augmentation.get("horizontal_flip", 0.0)
        if flip_prob:
            transforms.append(v2.RandomHorizontalFlip(p=flip_prob))
        jitter = augmentation.get("color_jitter", 0.0)
        if jitter:
            transforms.append(
                v2.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter)
            )
        if augmentation.get("random_crop", False):
            transforms.append(v2.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
        # Убираем рамки, которые испортились после случайных преобразований.
        transforms.append(v2.SanitizeBoundingBoxes())

    transforms.extend(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
    return v2.Compose(transforms)
