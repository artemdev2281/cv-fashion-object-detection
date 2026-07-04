# Модель EfficientDet.
#
# Берём её из сторонней библиотеки effdet: backbone tf_efficientdet_d0 с
# предобученными на COCO весами, а голову переделываем под наши 11 классов.
#
# У effdet есть несколько особенностей, из-за которых он работает не так, как
# модели из torchvision. Их нужно учесть, иначе модель обучится неправильно:
#
# - Номера классов. В самом проекте классы идут с 0 (0..10, num_classes=11,
#   без сдвига под фон). Но внутри effdet в двух местах нужно прибавлять/вычитать 1:
#     1) при обучении рамкам в таргете класс надо давать со сдвигом +1 (1..11),
#        потому что effdet использует 0 как "тут ничего нет" и потом сам вычитает 1;
#     2) при предсказании effdet, наоборот, отдаёт классы со сдвигом +1, поэтому
#        в адаптере ниже мы вычитаем 1 обратно.
#   Без первого сдвига класс 0 у нас просто пропадал, а остальные классы
#   учились со сдвигом - это проверено на реальном обучении.
# - Формат рамок. При обучении effdet ждёт рамки в порядке yxyx (а не xyxy, как
#   в torchvision). Предсказания он уже отдаёт в xyxy, там переводить не надо.
# - Формат картинки. На вход нужна квадратная картинка фиксированного размера
#   (обычно 512). Приводим её к квадрату через letterbox: меняем размер с
#   сохранением пропорций и добавляем чёрные поля. Так объекты не искажаются, и
#   потом легко вернуть рамки к исходным координатам по одному числу-масштабу.
# - img_size задаётся в порядке (ширина, высота), а не (высота, ширина).

from __future__ import annotations

from typing import Optional, Sequence

import torch

# Архитектура по умолчанию - самая лёгкая в семействе EfficientDet, чтобы
# хватало видеокарты Tesla T4 в Kaggle.
DEFAULT_ARCHITECTURE = "tf_efficientdet_d0"
# Размер картинки, стандартный для d0.
DEFAULT_IMAGE_SIZE = 512

# Числа ImageNet для нормализации входа. В отличие от Faster R-CNN и SSD,
# которые нормализуют картинку сами внутри, effdet этого не делает - поэтому
# нормализуем вручную здесь. Без этого предобученный backbone получал бы
# картинку не в том виде, к которому привык, и работал бы хуже.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_model(num_classes: int, config: dict | None = None, architecture: str = DEFAULT_ARCHITECTURE):
    """Создать модель EfficientDet под наше число классов.

    num_classes - число классов одежды (11), без сдвига под фон (см. комментарий
    в начале файла).
    config - настройки; из них может браться размер картинки.
    architecture - какую архитектуру effdet использовать (по умолчанию d0).

    Возвращает модель EfficientDet с новой головой под наши классы. Backbone
    предобучен на COCO, а голова учится с нуля.
    """
    try:
        from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
        from effdet.efficientdet import HeadNet
    except ImportError as error:  # библиотека может быть не установлена
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
    except TypeError:  # в старых версиях effdet нужен ещё один аргумент
        net.class_net = HeadNet(
            effdet_config, num_outputs=num_classes,
            norm_kwargs=dict(eps=1e-3, momentum=1e-2),
        )

    bench = DetBenchTrain(net, create_labeler=True)
    bench.image_size = image_size  # запомним размер, пригодится дальше
    return bench


def _letterbox(image: torch.Tensor, size: int) -> tuple[torch.Tensor, float]:
    """Изменить размер картинки с сохранением пропорций и дополнить до квадрата.

    image - картинка [C, H, W] со значениями от 0 до 1. Возвращаем квадратную
    картинку размера size и число scale = size / max(H, W). По scale потом
    легко пересчитать рамки обратно в исходные координаты.

    Заодно нормализуем картинку (числами ImageNet), потому что effdet сам этого
    не делает. Нормализуем только саму картинку, а чёрные поля оставляем нулями.
    Одна и та же функция используется и при обучении, и при предсказании.
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
    """Обёртка над нашим CocoDetectionDataset, чтобы он подходил для effdet.

    Берёт обычные (картинка, разметка) из нашего датасета и переделывает их в
    тот вид, который нужен effdet: картинку прогоняет через letterbox, рамки
    переводит в порядок yxyx, а номера классов сдвигает на +1 (это обязательно,
    см. комментарий в начале файла).
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

        boxes_xyxy = target["boxes"] * scale  # масштабируем рамки под letterbox
        boxes_yxyx = boxes_xyxy[:, [1, 0, 3, 2]]  # xyxy -> yxyx, как ждёт effdet
        orig_h, orig_w = image.shape[-2], image.shape[-1]

        effdet_target = {
            "bbox": boxes_yxyx,
            # +1 обязателен (см. комментарий в начале файла): без него класс 0
            # у нас пропадал бы, а остальные классы учились бы со сдвигом.
            "cls": (target["labels"] + 1).to(torch.float32),
            "img_scale": torch.tensor(1.0 / scale, dtype=torch.float32),
            # размер картинки в порядке (ширина, высота) - так ждёт effdet
            "img_size": torch.tensor([orig_w, orig_h], dtype=torch.float32),
            "image_id": target["image_id"],
        }
        return canvas, effdet_target


def effdet_collate_fn(batch):
    """Как собирать батч для EfficientDetCocoDataset.

    После letterbox все картинки одного размера, поэтому их можно просто
    сложить в один тензор. А рамки и классы у картинок разной длины, поэтому их
    оставляем списками - effdet именно так их и ждёт.
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
    """Создать DataLoader над EfficientDetCocoDataset."""
    dataset = EfficientDetCocoDataset(
        ann_file, data_root, image_size=image_size, subset_size=subset_size,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=effdet_collate_fn,
    )


class EfficientDetPredictAdapter(torch.nn.Module):
    """Адаптер для предсказаний EfficientDet, чтобы их можно было оценить.

    Принимает список картинок (значения от 0 до 1, обычные из нашего датасета) и
    возвращает предсказания в том виде, который ждёт evaluate_coco_detector:
    рамки в xyxy в исходных координатах, уверенности и номера классов (с 0).

    Порог score_thr низкий (0.05) специально - чтобы при подсчёте mAP учитывались
    все предсказания. Обычный порог 0.5 применяется отдельно уже при оценке.

    Классы на выходе должны быть 0..10, а не 1..11 - здесь мы вычитаем обратно
    тот +1, который effdet прибавляет сам.
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
            # размер в порядке (ширина, высота) - см. комментарий в начале файла
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
            labels = det[:, 5].long() - 1  # вычитаем тот +1, что прибавил effdet
            labels = labels.clamp(min=0)
            outputs.append({
                "boxes": det[:, :4],
                "scores": det[:, 4],
                "labels": labels,
            })
        return outputs
