# Датасет в формате COCO для моделей из torchvision (Faster R-CNN, SSD).
#
# Читает наши COCO-аннотации (data/processed/coco/{split}.json) и выдаёт пары
# (картинка, разметка) в том виде, который ждёт torchvision.
#
# Пара важных моментов:
# - Сдвиг меток на +1. В наших файлах классы идут с 0 (0..10). А в torchvision
#   класс 0 - это фон, поэтому при подаче в модель прибавляем label_offset (=1),
#   и классы становятся 1..11, а всего классов у модели 12. Обратно вычитаем
#   при подсчёте метрик (см. src/evaluation/metrics.py).
# - Нормализацию тут НЕ делаем, отдаём только пиксели от 0 до 1. Faster R-CNN
#   и SSD нормализуют вход сами внутри, иначе получилось бы два раза.
# - Путь к картинке - это data_root + file_name из аннотаций.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image


def collate_fn(batch):
    """Как собирать батч: картинки и разметка бывают разного размера.

    Обычный способ склейки батча тут не подходит (у картинок разное число
    объектов и разные размеры), поэтому просто складываем их в кортеж списков.
    """
    return tuple(zip(*batch))


class CocoDetectionDataset(torch.utils.data.Dataset):
    """Датасет в формате COCO для моделей torchvision.

    ann_file - путь к json-файлу части данных (например, coco/train.json).
    data_root - папка с картинками, к ней прибавляется file_name из аннотаций.
    label_offset - на сколько сдвигать номера классов (по умолчанию 1, потому
    что класс 0 занят под фон).
    subset_size - если задано, взять только первые N картинок (для быстрой
    проверки).
    """

    def __init__(
        self,
        ann_file: str | Path,
        data_root: str | Path,
        label_offset: int = 1,
        subset_size: Optional[int] = None,
    ) -> None:
        from pycocotools.coco import COCO

        self.data_root = Path(data_root)
        self.label_offset = label_offset
        self.coco = COCO(str(ann_file))
        ids = sorted(self.coco.imgs.keys())
        if subset_size is not None:
            ids = ids[:subset_size]
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        from torchvision.transforms import functional as F

        coco = self.coco
        image_id = self.ids[index]
        image_info = coco.loadImgs(image_id)[0]
        image_path = self.data_root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:  # пропускаем плохие рамки
                continue
            boxes.append([x, y, x + w, y + h])  # из формата xywh в xyxy
            labels.append(ann["category_id"] + self.label_offset)
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return F.to_tensor(image), target


def build_loader(
    ann_file: str | Path,
    data_root: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    label_offset: int = 1,
    subset_size: Optional[int] = None,
):
    """Создать DataLoader над нашим COCO-датасетом."""
    dataset = CocoDetectionDataset(
        ann_file, data_root, label_offset=label_offset, subset_size=subset_size
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader
