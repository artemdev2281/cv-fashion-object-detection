"""Модель DETR (детектор на основе архитектуры трансформера).

Реализация через HuggingFace ``transformers``: ``DetrForObjectDetection``
(``facebook/detr-resnet-50``, предобучена на COCO) + ``DetrImageProcessor``
для препроцессинга/постпроцессинга. Интерфейс согласован с остальными
детекторами проекта, но, в отличие от них, у DETR полностью своя конвенция
данных:

* **Индексация классов.** У DETR нет отдельного класса «фон» со сдвигом +1,
  как в torchvision (см. :mod:`src.dataset.coco_dataset`): ``num_labels=11``,
  метки 0-based (как ``category_id`` в ``classes.json``), «no-object»
  моделируется отдельным обучаемым слотом внутри декодера DETR.
  ``label_offset=0`` при вызове
  :func:`src.evaluation.metrics.evaluate_coco_detector`.
* **Формат таргета.** ``DetrImageProcessor`` сам строит таргет из COCO-style
  аннотаций (``{"image_id", "annotations": [...]}``) и переводит рамки в
  нормализованный ``cxcywh`` — таргет собирается процессором, не руками.
* **Обучение.** Раздельный learning rate для backbone (меньше) и остальной
  сети (transformer + головы) — стандартная практика fine-tuning DETR;
  оптимизатор AdamW. Форвард возвращает уже суммарный ``outputs.loss``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

#: Предобученный чекпоинт DETR (ResNet-50 backbone, обучен на COCO).
DEFAULT_CHECKPOINT = "facebook/detr-resnet-50"


def build_model(num_classes: int, config: dict | None = None, checkpoint: str = DEFAULT_CHECKPOINT):
    """Создать DETR (``DetrForObjectDetection``) под ``num_classes``.

    Параметры
    ---------
    num_classes:
        Число категорий одежды (11), БЕЗ фонового +1 — см. модульный докстринг.
    config:
        Конфигурация эксперимента (для совместимости интерфейса; гиперпараметры
        обучения передаются в цикл обучения, а не в конструктор).
    checkpoint:
        Имя предобученного чекпоинта HuggingFace Hub.

    Возвращает
    ----------
    ``transformers.DetrForObjectDetection`` с классификационной головой,
    переинициализированной под ``num_classes`` (несовпадающий размер головы с
    оригинальным COCO-чекпоинтом (91 класс) допускается через
    ``ignore_mismatched_sizes=True``).
    """
    try:
        from transformers import DetrForObjectDetection
    except ImportError as error:  # pragma: no cover - зависит от среды
        raise ImportError(
            "Для DETR требуется пакет transformers (pip install transformers). "
            "См. ячейку установки зависимостей в "
            "notebooks/train_efficientdet_detr.ipynb."
        ) from error

    model = DetrForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def build_processor(checkpoint: str = DEFAULT_CHECKPOINT, image_size: int = 800):
    """Создать ``DetrImageProcessor`` (препроцессинг/постпроцессинг DETR).

    ``image_size`` — сторона, к которой процессор приводит меньшую сторону
    изображения (стандартный ресайз DETR с сохранением соотношения сторон;
    процессор сам делает паддинг батча под общий размер через ``.pad``).
    """
    from transformers import DetrImageProcessor

    return DetrImageProcessor.from_pretrained(checkpoint, size={"shortest_edge": image_size, "longest_edge": image_size * 2})


def _boxes_xyxy_to_coco_annotations(target: dict) -> list[dict]:
    """Собрать COCO-style annotations из таргета ``CocoDetectionDataset`` (label_offset=0).

    ``DetrImageProcessor`` ожидает на вход COCO-аннотации (``bbox`` в ``xywh``,
    ``category_id``, ``area``), а не готовые тензоры ``boxes``/``labels`` —
    формирует нормализованный ``cxcywh``-таргет сам.
    """
    boxes = target["boxes"].tolist()
    labels = target["labels"].tolist()
    areas = target["area"].tolist()
    annotations = []
    for index, (box, label, area) in enumerate(zip(boxes, labels, areas)):
        x1, y1, x2, y2 = box
        annotations.append({
            "id": index,
            "image_id": int(target["image_id"].item()),
            "category_id": int(label),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": float(area),
            "iscrowd": 0,
        })
    return annotations


class DetrCocoDataset(torch.utils.data.Dataset):
    """Адаптер над :class:`src.dataset.coco_dataset.CocoDetectionDataset` под DETR.

    Переиспользует уже готовый класс проекта как источник изображений и
    таргетов (``label_offset=0``, та же логика отбора подвыборки
    ``sorted(image_id)[:subset_size]``), и прогоняет пару
    (изображение, COCO-аннотации) через ``DetrImageProcessor``, который сам
    строит нормализованный таргет DETR.
    """

    def __init__(self, ann_file, data_root, processor, subset_size: Optional[int] = None) -> None:
        from src.dataset.coco_dataset import CocoDetectionDataset

        self.inner = CocoDetectionDataset(
            ann_file, data_root, label_offset=0, subset_size=subset_size,
        )
        self.processor = processor
        self.coco = self.inner.coco
        self.ids = self.inner.ids

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        from torchvision.transforms import functional as F

        image, target = self.inner[index]
        pil_image = F.to_pil_image(image)
        annotations = _boxes_xyxy_to_coco_annotations(target)

        encoded = self.processor(
            images=pil_image,
            annotations={"image_id": int(target["image_id"].item()), "annotations": annotations},
            return_tensors="pt",
        )
        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"][0]
        return pixel_values, labels


def build_detr_collate_fn(processor):
    """collate_fn для ``DetrCocoDataset``: ручной паддинг батча (без ``processor.pad``).

    Изображения после ``DetrImageProcessor`` имеют разный размер (ресайз с
    сохранением аспекта под кратчайшую/длиннейшую сторону), поэтому батч нужно
    привести к общему размеру. Паддинг сделан вручную (``F.pad`` до макс. H/W
    в батче + ``pixel_mask`` из единиц/нулей), а НЕ через ``processor.pad`` —
    сигнатура этого метода нестабильна между версиями ``transformers``
    (проверено: в установленной здесь версии ``pad`` работает с ОДНИМ
    изображением и параметром ``padded_size``, без batched
    ``return_tensors="pt"`` из классических туториалов по DETR) — собственная
    реализация не зависит от версии.

    ``processor`` принимается для единообразия сигнатуры (не используется
    здесь), т. к. остаётся востребованным в остальных местах модуля.
    """

    def collate_fn(batch):
        import torch.nn.functional as nn_functional

        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        max_h = max(pv.shape[-2] for pv in pixel_values)
        max_w = max(pv.shape[-1] for pv in pixel_values)

        padded_values, pixel_masks = [], []
        for pv in pixel_values:
            _, height, width = pv.shape
            padded = nn_functional.pad(pv, (0, max_w - width, 0, max_h - height), value=0.0)
            mask = torch.zeros((max_h, max_w), dtype=torch.long)
            mask[:height, :width] = 1
            padded_values.append(padded)
            pixel_masks.append(mask)

        return {
            "pixel_values": torch.stack(padded_values),
            "pixel_mask": torch.stack(pixel_masks),
            "labels": labels,
        }

    return collate_fn


def build_detr_loader(
    ann_file, data_root, processor, batch_size: int, shuffle: bool,
    num_workers: int = 2, subset_size: Optional[int] = None,
):
    """Собрать ``DataLoader`` над :class:`DetrCocoDataset`."""
    dataset = DetrCocoDataset(ann_file, data_root, processor, subset_size=subset_size)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=build_detr_collate_fn(processor),
    )


class DetrPredictAdapter(torch.nn.Module):
    """Тонкий инференс-адаптер DETR под контракт ``evaluate_coco_detector``.

    В ``eval``-режиме принимает список тензоров-изображений произвольного
    размера в ``[0, 1]`` (как отдаёт обычный
    :class:`src.dataset.coco_dataset.CocoDetectionDataset`, используемый БЕЗ
    изменений для eval-loader'а) и возвращает ``List[Dict]`` с ``boxes``
    (``xyxy`` в исходных пиксельных координатах — процессор пересчитывает их
    сам через ``target_sizes``), ``scores`` и ``labels`` (0-based).

    ``threshold`` низкий (0.05) — чтобы COCOeval видел полную кривую
    precision/recall при подсчёте mAP; операционная точка P/R внутри
    ``evaluate_coco_detector`` отдельно считается на 0.5.
    """

    def __init__(self, model, processor, threshold: float = 0.05) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, images: Sequence[torch.Tensor]) -> list[dict]:
        from torchvision.transforms import functional as F

        device = next(self.model.parameters()).device
        pil_images = [F.to_pil_image(image.cpu()) for image in images]
        encoded = self.processor(images=pil_images, return_tensors="pt").to(device)

        self.model.eval()
        outputs = self.model(**encoded)

        target_sizes = torch.tensor([[image.shape[-2], image.shape[-1]] for image in images], device=device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes,
        )
        return [
            {"boxes": result["boxes"], "scores": result["scores"], "labels": result["labels"]}
            for result in results
        ]
