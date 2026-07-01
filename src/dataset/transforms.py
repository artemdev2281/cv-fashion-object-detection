"""Предобработка изображений: нормализация и аугментации.

Модуль задаёт преобразования, применяемые к изображениям на этапе обучения:
приведение к целевому размеру (resize), нормализацию пикселей и аугментации.

Важно про разделение ответственности между моделями:

* **YOLOv8 (baseline)** выполняет resize (letterbox), нормализацию и
  аугментации (mosaic, horizontal flip, HSV-искажения и др.) **встроенно** —
  через аргументы обучения Ultralytics. Поэтому изображения на диск
  сохраняются в исходном виде, а не преаугментированными; параметры
  аугментации для YOLOv8 передаются функцией :func:`yolo_augmentation_args`.
* **Faster R-CNN, SSD, EfficientDet, DETR** (torchvision-совместимые) получают
  преобразования через :func:`build_transforms`.

Аугментации применяются **только к train**; для val и test выполняется лишь
детерминированный resize и нормализация — чтобы не искажать оценку качества.
"""

from __future__ import annotations

from typing import Sequence

#: Средние и стандартные отклонения ImageNet (для нормализации пикселей).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def yolo_augmentation_args(config: dict) -> dict:
    """Сформировать параметры аугментации YOLOv8 (Ultralytics) из конфигурации.

    Возвращает словарь аргументов обучения Ultralytics, соответствующих секции
    ``augmentation`` конфигурации. Аугментации YOLOv8 применяются только к
    обучающей выборке самим фреймворком.
    """
    augmentation = config.get("augmentation", {})
    return {
        "fliplr": augmentation.get("horizontal_flip", 0.0),
        "hsv_s": augmentation.get("color_jitter", 0.0),
        "hsv_v": augmentation.get("color_jitter", 0.0),
        "mosaic": augmentation.get("mosaic", 0.0),
        # crop у Ultralytics задаётся через scale; включаем при random_crop.
        "scale": 0.5 if augmentation.get("random_crop", False) else 0.0,
    }


def build_transforms(config: dict, split: str):
    """Построить преобразования torchvision для не-YOLO моделей.

    Использует API ``torchvision.transforms.v2``, корректно преобразующее
    вместе с изображением и ограничивающие рамки (при передаче цели как
    ``tv_tensors.BoundingBoxes``). Аугментации добавляются только для ``train``.

    Параметры
    ---------
    config:
        Конфигурация эксперимента (секции ``training`` и ``augmentation``).
    split:
        Имя сплита (``train`` / ``val`` / ``test``).

    Возвращает
    ----------
    Объект ``torchvision.transforms.v2.Compose``.
    """
    try:
        import torch
        from torchvision.transforms import v2
    except ImportError as error:  # pragma: no cover - зависит от среды выполнения
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
        # Отбрасывание рамок, выродившихся после геометрических преобразований.
        transforms.append(v2.SanitizeBoundingBoxes())

    transforms.extend(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
    return v2.Compose(transforms)
