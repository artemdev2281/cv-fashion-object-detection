"""Модель EfficientDet (детектор, оптимизированный по эффективности).

Реализация через сторонний пакет ``effdet`` (rwightman/efficientdet-pytorch):
backbone ``tf_efficientdet_d0`` + предобученные веса (COCO), голова
переинициализируется под ``num_classes`` категорий проекта.

Конвенции ``effdet``, отличающие его от torchvision-детекторов проекта
(Faster R-CNN/SSD) — см. также :mod:`src.training.train`:

* **Индексация классов — ДВА независимых +1, оба внутренние детали ``effdet``,
  а не публичный контракт проекта.** На уровне проекта (``build_model``,
  ``evaluate_coco_detector``) классы 0-based (0..10, ``num_classes=11``,
  ``label_offset=0``, БЕЗ фонового сдвига, как у torchvision) — это не
  меняется. Но ВНУТРИ ``effdet`` есть два разных технических +1, которые
  нужно применять на границе с библиотекой:

  1. **Вход в ``AnchorLabeler`` (обучение).** Подтверждено чтением
     ``effdet/anchors.py::label_anchors``/``batch_label_anchors`` (строка
     ``# class labels start from 1 and the background class = -1``
     ``cls_targets = (cls_targets - 1).long()``): библиотека использует
     raw-класс ``0`` как служебное значение "анкор не сопоставлен ни с одним
     объектом" и лишь потом сама вычитает 1. Значит, ``gt_classes``,
     переданные в таргет, ДОЛЖНЫ быть 1-based (1..11) — иначе анкоры,
     сопоставленные с классом 0 проекта, стирались бы в фон, а остальные
     классы обучались бы со сдвигом -1 (именно так и происходило до
     исправления — см. ``EfficientDetCocoDataset.__getitem__``, ``cls`` там
     сдвинут на +1).
  2. **Выход ``DetBenchPredict`` (инференс).** Отдельно от п. 1, подтверждено
     чтением ``effdet/anchors.py::generate_detections`` (строка
     ``classes = classes[...] + 1  # back to class idx with background
     class = 0``): сырые предсказания приходят со своим +1, который
     компенсируется вычитанием 1 в инференс-адаптере ниже.

  Оба сдвига проверены НЕ только чтением исходника, но и прогоном полного
  цикла обучения на реальных данных Fashionpedia (не только на синтетике) —
  без п. 1 сеть обучалась с систематическим сдвигом классов на -1 и полным
  стиранием класса 0, что и показал реальный прогон (см. память проекта).
* **Формат bbox таргета.** При обучении ожидает рамки в порядке ``yxyx``
  (не ``xyxy``, как в torchvision) — подтверждено по построению анкоров в
  ``effdet/anchors.py`` (``[y0, x0, y1, x1]``). Выходные детекции
  ``DetBenchPredict``, наоборот, уже в ``xyxy`` — конвертация не нужна.
* **Формат изображения.** Вход — квадратный тензор фиксированного размера
  (``image_size``, обычно 512 для d0). Ресайз выполняется letterbox'ом
  (сохранение соотношения сторон + паддинг нулями до квадрата), а не простым
  stretch-resize — так сохраняется корректная геометрия объектов и есть
  единственный скалярный коэффициент масштаба на изображение, которым можно
  корректно вернуть предсказанные рамки в исходные пиксельные координаты.
* **`img_size` — порядок (W, H), не (H, W).** Подтверждено чтением
  ``effdet/anchors.py::clip_boxes_xyxy``, которая дублирует ``size`` в
  ``[W, H, W, H]`` и клипует им ``[x1, y1, x2, y2]``.

Все перечисленные конвенции проверены как чтением исходников установленной
версии ``effdet``, так и локальным прогоном полного цикла (build_model →
train-шаг → inference-адаптер) на синтетических данных перед сдачей — но
финальная проверка на реальных данных Fashionpedia всё равно нужна, т.к. поведение на необученной сети не гарантирует
отсутствие тонких ошибок при реальном обучении.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

#: Архитектура backbone по умолчанию — самая лёгкая в семействе EfficientDet,
#: под бюджет Tesla T4 и GPU-квоту Kaggle (см. память проекта).
DEFAULT_ARCHITECTURE = "tf_efficientdet_d0"
#: Разрешение входа, стандартное для d0.
DEFAULT_IMAGE_SIZE = 512

#: ImageNet mean/std для нормализации входа. В ОТЛИЧИЕ от torchvision-детекторов
#: (Faster R-CNN/SSD нормализуют вход ВНУТРИ себя через GeneralizedRCNNTransform,
#: поэтому CocoDetectionDataset отдаёт [0,1] без нормализации), модель ``effdet``
#: нормализацию внутри НЕ делает — её штатно применяет пайплайн
#: ``effdet.data.transforms``, который здесь заменён своим адаптером. Без этой
#: нормализации предобученный на ImageNet backbone получает вход не в той
#: статистике, в которой обучался (ждёт ~[-2,2], получал бы [0,1]).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_model(num_classes: int, config: dict | None = None, architecture: str = DEFAULT_ARCHITECTURE):
    """Создать EfficientDet (обёртка ``DetBenchTrain``) под ``num_classes``.

    Параметры
    ---------
    num_classes:
        Число категорий одежды (11), БЕЗ фонового +1 — см. модульный докстринг.
    config:
        Конфигурация эксперимента; ``training.efficientdet_image_size``
        (если задан) переопределяет :data:`DEFAULT_IMAGE_SIZE`.
    architecture:
        Имя архитектуры ``effdet`` (по умолчанию ``tf_efficientdet_d0``).

    Возвращает
    ----------
    ``effdet.DetBenchTrain``, оборачивающий предобученную сеть с новой головой
    классификации (``net.class_net``) под ``num_classes``. Веса backbone и
    BiFPN — предобученные на COCO; голова — обучается с нуля.
    """
    try:
        from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
        from effdet.efficientdet import HeadNet
    except ImportError as error:  # pragma: no cover - зависит от среды
        raise ImportError(
            "Для EfficientDet требуется пакет effdet (pip install effdet). "
            "См. ячейку установки зависимостей в "
            "notebooks/train_efficientdet_detr.ipynb."
        ) from error

    image_size = int(
        (config or {}).get("training", {}).get("efficientdet_image_size", DEFAULT_IMAGE_SIZE)
    )

    effdet_config = get_efficientdet_config(architecture)
    effdet_config.num_classes = num_classes
    effdet_config.image_size = (image_size, image_size)

    net = EfficientDet(effdet_config, pretrained_backbone=True)
    try:
        net.class_net = HeadNet(effdet_config, num_outputs=num_classes)
    except TypeError:  # старые версии effdet требуют явный norm_kwargs
        net.class_net = HeadNet(
            effdet_config, num_outputs=num_classes,
            norm_kwargs=dict(eps=1e-3, momentum=1e-2),
        )

    bench = DetBenchTrain(net, create_labeler=True)
    bench.image_size = image_size  # для удобства чтения адаптером/тренером
    return bench


def _letterbox(image: torch.Tensor, size: int) -> tuple[torch.Tensor, float]:
    """Ресайз с сохранением соотношения сторон + паддинг нулями до квадрата.

    ``image`` — тензор ``[C, H, W]`` в ``[0, 1]``. Возвращает
    ``(canvas [C, size, size], scale)``, где ``scale = size / max(H, W)`` —
    единственный коэффициент, связывающий координаты в letterbox-пространстве
    с исходным изображением (``resized = original * scale``).

    Изображение нормализуется ImageNet mean/std (см. :data:`_IMAGENET_MEAN`),
    т.к. модель ``effdet`` не нормализует вход сама. Нормализуется ТОЛЬКО
    ресайзнутая часть, паддинг остаётся 0 (как в штатном пайплайне effdet —
    паддинг после нормализации). Применяется одинаково в train (датасет) и
    inference (адаптер), т.к. оба идут через ``_letterbox``.
    """
    from torchvision.transforms import functional as F

    channels, height, width = image.shape
    scale = size / max(height, width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    resized = F.resize(image, [new_h, new_w], antialias=True)
    resized = (resized - _IMAGENET_MEAN.to(resized)) / _IMAGENET_STD.to(resized)
    canvas = image.new_zeros((channels, size, size))
    canvas[:, :new_h, :new_w] = resized
    return canvas, scale


class EfficientDetCocoDataset(torch.utils.data.Dataset):
    """Адаптер над :class:`src.dataset.coco_dataset.CocoDetectionDataset` под ``effdet``.

    Переиспользует уже готовый и провалидированный класс проекта (в т. ч. его
    логику отбора подвыборки ``sorted(image_id)[:subset_size]``) как источник
    ``(image, target_torchvision)`` и только конвертирует изображение/таргет в
    формат, ожидаемый ``DetBenchTrain``: letterbox-ресайз изображения,
    рамки — в ``yxyx`` в масштабе letterbox-пространства, метки — со сдвигом
    +1 (см. модульный докстринг, п. 1 — обязателен для ``AnchorLabeler``).
    """

    def __init__(self, ann_file, data_root, image_size: int = DEFAULT_IMAGE_SIZE,
                 subset_size: Optional[int] = None) -> None:
        from src.dataset.coco_dataset import CocoDetectionDataset

        self.inner = CocoDetectionDataset(
            ann_file, data_root, label_offset=0, subset_size=subset_size,
        )
        self.image_size = image_size
        self.coco = self.inner.coco
        self.ids = self.inner.ids

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        image, target = self.inner[index]
        canvas, scale = _letterbox(image, self.image_size)

        boxes_xyxy = target["boxes"] * scale  # letterbox top-left aligned, без сдвига
        boxes_yxyx = boxes_xyxy[:, [1, 0, 3, 2]]
        orig_h, orig_w = image.shape[-2], image.shape[-1]

        effdet_target = {
            "bbox": boxes_yxyx,
            # +1 — ОБЯЗАТЕЛЕН для входа в AnchorLabeler (см. модульный докстринг):
            # effdet/anchors.py использует raw-класс 0 как служебное значение
            # "анкор не сопоставлен", а затем сам вычитает 1 (0-based класс
            # проекта БЕЗ +1 здесь привёл бы к тому, что совпадения с классом 0
            # стирались бы в фон, а остальные классы обучались бы со сдвигом -1).
            "cls": (target["labels"] + 1).to(torch.float32),
            "img_scale": torch.tensor(1.0 / scale, dtype=torch.float32),
            # порядок (W, H) — подтверждено чтением effdet/anchors.py::clip_boxes_xyxy,
            # которая дублирует size в [W, H, W, H] и клипует [x1, y1, x2, y2] им же.
            "img_size": torch.tensor([orig_w, orig_h], dtype=torch.float32),
            "image_id": target["image_id"],
        }
        return canvas, effdet_target


def effdet_collate_fn(batch):
    """collate_fn для ``EfficientDetCocoDataset``.

    Изображения после letterbox — фиксированного размера, стандартный
    ``torch.stack`` работает. ``bbox``/``cls`` — переменной длины на
    изображение, поэтому остаются списками тензоров (так их и ожидает
    ``DetBenchTrain.forward`` при ``create_labeler=True``); ``img_scale`` и
    ``img_size`` — батчатся в тензоры.
    """
    images = torch.stack([item[0] for item in batch])
    target = {
        "bbox": [item[1]["bbox"] for item in batch],
        "cls": [item[1]["cls"] for item in batch],
        "img_scale": torch.stack([item[1]["img_scale"] for item in batch]),
        "img_size": torch.stack([item[1]["img_size"] for item in batch]),
    }
    return images, target


def build_effdet_loader(
    ann_file, data_root, batch_size: int, shuffle: bool,
    image_size: int = DEFAULT_IMAGE_SIZE, num_workers: int = 2,
    subset_size: Optional[int] = None,
):
    """Собрать ``DataLoader`` над :class:`EfficientDetCocoDataset`."""
    dataset = EfficientDetCocoDataset(
        ann_file, data_root, image_size=image_size, subset_size=subset_size,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=effdet_collate_fn,
    )


class EfficientDetPredictAdapter(torch.nn.Module):
    """Тонкий инференс-адаптер EfficientDet под контракт ``evaluate_coco_detector``.

    В ``eval``-режиме принимает список тензоров-изображений произвольного
    размера в ``[0, 1]`` (как отдаёт обычный
    :class:`src.dataset.coco_dataset.CocoDetectionDataset`, используемый БЕЗ
    изменений для eval-loader'а) и возвращает ``List[Dict]`` с ``boxes``
    (``xyxy`` в исходных пиксельных координатах), ``scores`` и ``labels``
    (0-based), как того ожидает
    :func:`src.evaluation.metrics.evaluate_coco_detector`.

    ``score_thr`` низкий (0.05) специально — чтобы COCOeval видел полную
    кривую precision/recall при подсчёте mAP (как у torchvision-детекторов,
    у которых внутренний порог по умолчанию тоже низкий); операционная
    точка P/R внутри ``evaluate_coco_detector`` отдельно считается на 0.5.

    **Проверить на smoke-тесте:** диапазон ``labels`` должен быть 0..10 (не
    1..11) — см. предупреждение про сдвиг +1 в модульном докстринге.
    """

    def __init__(self, train_bench, image_size: int = DEFAULT_IMAGE_SIZE, score_thr: float = 0.05) -> None:
        super().__init__()
        from effdet import DetBenchPredict

        self.predict_bench = DetBenchPredict(train_bench.model)
        self.image_size = image_size
        self.score_thr = score_thr

    @torch.no_grad()
    def forward(self, images: Sequence[torch.Tensor]) -> list[dict]:
        device = images[0].device
        canvases, scales, sizes = [], [], []
        for image in images:
            canvas, scale = _letterbox(image, self.image_size)
            canvases.append(canvas)
            scales.append(1.0 / scale)
            # (W, H) — см. комментарий в EfficientDetCocoDataset.__getitem__.
            sizes.append([image.shape[-1], image.shape[-2]])

        batch = torch.stack(canvases).to(device)
        img_info = {
            "img_scale": torch.tensor(scales, dtype=torch.float32, device=device),
            "img_size": torch.tensor(sizes, dtype=torch.float32, device=device),
        }

        self.predict_bench.eval()
        detections = self.predict_bench(batch, img_info)  # [B, max_det, 6]: x1,y1,x2,y2,score,class

        outputs = []
        for det in detections:
            keep = det[:, 4] >= self.score_thr
            det = det[keep]
            labels = det[:, 5].long() - 1  # см. докстринг: effdet сдвигает класс на +1
            labels = labels.clamp(min=0)
            outputs.append({
                "boxes": det[:, :4],
                "scores": det[:, 4],
                "labels": labels,
            })
        return outputs
