"""COCO-датасет для torchvision detection моделей (Faster R-CNN, SSD).

Общий на обе модели класс: читает COCO-аннотации проекта
(``data/processed/coco/{split}.json``) и отдаёт пары ``(image, target)`` в
формате, который ожидает ``torchvision.models.detection``.

Ключевые тонкости
-----------------
* **Сдвиг меток на +1.** В COCO-файлах проекта ``category_id`` идут от 0
  (shoe=0 … tights=10). В torchvision detection API класс **0 зарезервирован
  под фон**, поэтому при подаче в модель метки сдвигаются на ``label_offset``
  (=1): классы становятся 1..11, а ``num_classes`` модели = 12. Обратный сдвиг
  выполняется на этапе оценки метрик (см. :mod:`src.evaluation.metrics`).
* **Нормализация.** Отдаётся только ``to_tensor`` (пиксели в ``[0, 1]``).
  Детекторы torchvision (Faster R-CNN, SSD) нормализуют вход **внутри себя**
  (``GeneralizedRCNNTransform``), поэтому ручная нормализация здесь НЕ
  выполняется — иначе была бы двойная.
* **Путь к изображениям** строится как ``data_root / file_name`` (в COCO JSON
  ``file_name`` — относительный путь вида ``images/train/xxxx.jpg``).
  ``data_root`` передаётся явно (как ``DATA_ROOT`` в ноутбуке), не хардкодится.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image


def collate_fn(batch):
    """collate_fn для detection: изображения и таргеты переменного размера.

    Стандартный ``default_collate`` не работает (разное число объектов и разные
    размеры изображений), поэтому batch превращается в кортеж списков — как
    принято в torchvision detection tutorials.
    """
    return tuple(zip(*batch))


class CocoDetectionDataset(torch.utils.data.Dataset):
    """Датасет COCO-формата для torchvision detection.

    Параметры
    ---------
    ann_file:
        Путь к COCO JSON сплита (``.../coco/train.json`` и т. п.).
    data_root:
        Корень подготовленных данных (``data/processed`` или его путь в Kaggle).
        Складывается с относительным ``file_name`` из аннотаций.
    label_offset:
        Сдвиг меток при подаче в модель (по умолчанию ``1`` — резерв класса 0
        под фон).
    subset_size:
        Если задано — берутся первые ``subset_size`` изображений (для
        smoke-теста).
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
            if w <= 0 or h <= 0:  # защита от вырожденных рамок
                continue
            boxes.append([x, y, x + w, y + h])  # COCO xywh -> VOC xyxy
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
    """Собрать ``DataLoader`` c нужным ``collate_fn`` над COCO-датасетом."""
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
