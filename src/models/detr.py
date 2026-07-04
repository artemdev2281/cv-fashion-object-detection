# Модель DETR - детектор на основе трансформера.
#
# Берём её из библиотеки HuggingFace transformers: DetrForObjectDetection
# (facebook/detr-resnet-50, предобучена на COCO) плюс DetrImageProcessor,
# который готовит картинки на входе и разбирает предсказания на выходе.
# У DETR своя логика работы с данными:
#
# - Номера классов. В отличие от torchvision, тут нет отдельного класса "фон"
#   со сдвигом +1: классы идут с 0 (как в classes.json), а "тут ничего нет"
#   модель хранит отдельно внутри себя. Поэтому при оценке label_offset=0.
# - Разметку строит сам DetrImageProcessor из COCO-аннотаций, вручную её
#   собирать не нужно.
# - При обучении для backbone берём learning rate поменьше, чем для остальной
#   сети - это обычная практика для DETR. Оптимизатор - AdamW.

from __future__ import annotations

from typing import Optional, Sequence

import torch

# Предобученные веса DETR (backbone ResNet-50, обучена на COCO).
DEFAULT_CHECKPOINT = "facebook/detr-resnet-50"


def build_model(num_classes: int, config: dict | None = None, checkpoint: str = DEFAULT_CHECKPOINT):
    """Создать модель DETR под наше число классов.

    num_classes - число классов одежды (11), без сдвига под фон (см. комментарий
    в начале файла).
    config - настройки (нужен просто для единого вида функций).
    checkpoint - какие предобученные веса брать с HuggingFace.

    Возвращает модель DETR с новой головой под наши классы. У исходной модели
    было 91 класс, поэтому голову пересоздаём (ignore_mismatched_sizes=True).
    """
    try:
        from transformers import DetrForObjectDetection
    except ImportError as error:  # библиотека может быть не установлена
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
    """Создать DetrImageProcessor - он готовит картинки для DETR и разбирает
    её предсказания.

    image_size - размер, к которому процессор приводит меньшую сторону картинки
    (с сохранением пропорций).
    """
    from transformers import DetrImageProcessor

    return DetrImageProcessor.from_pretrained(checkpoint, size={"shortest_edge": image_size, "longest_edge": image_size * 2})


def _boxes_xyxy_to_coco_annotations(target: dict) -> list[dict]:
    """Собрать разметку в формате COCO из нашего таргета.

    DetrImageProcessor ждёт на вход именно COCO-аннотации (рамки в xywh,
    category_id, area), а не готовые тензоры - остальное он делает сам.
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
    """Обёртка над нашим CocoDetectionDataset, чтобы он подходил для DETR.

    Берёт картинки и разметку из нашего датасета и прогоняет их через
    DetrImageProcessor, который сам приводит их к нужному DETR виду.
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
    """Как собирать батч для DetrCocoDataset: дополняем картинки вручную.

    После обработки картинки в батче получаются разного размера, поэтому их надо
    дополнить до одинакового. Делаем это сами: добавляем нули по краям до
    максимальной ширины и высоты в батче и заодно строим pixel_mask (единицы там,
    где настоящая картинка, нули - где добавленные поля). Свой вариант надёжнее,
    потому что готовый processor.pad в разных версиях библиотеки работает
    по-разному.

    processor тут не используется, он в аргументах просто для единообразия.
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
    """Создать DataLoader над DetrCocoDataset."""
    dataset = DetrCocoDataset(ann_file, data_root, processor, subset_size=subset_size)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=build_detr_collate_fn(processor),
    )


class DetrPredictAdapter(torch.nn.Module):
    """Адаптер для предсказаний DETR, чтобы их можно было оценить.

    Принимает список картинок (значения от 0 до 1, обычные из нашего датасета) и
    возвращает предсказания в том виде, который ждёт evaluate_coco_detector:
    рамки в xyxy в исходных координатах (их пересчитывает сам процессор),
    уверенности и номера классов (с 0).

    Порог threshold низкий (0.05) специально - чтобы при подсчёте mAP учитывались
    все предсказания. Обычный порог 0.5 применяется отдельно уже при оценке.
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
