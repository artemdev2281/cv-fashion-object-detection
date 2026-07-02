"""Логика обучения моделей детектирования (реализовано для YOLOv8 baseline).

Модуль предоставляет процедуру обучения YOLOv8 через API Ultralytics и
приводит метрики к **единому контракту**, обязательному для всех пяти моделей
проекта (YOLOv8, Faster R-CNN, SSD, EfficientDet, DETR):

    {
        "map50":    float,   # mAP@0.5
        "map50_95": float,   # mAP@0.5:0.95 (методика COCO)
        "precision": float,
        "recall":   float,
        "f1":       float,   # 2*P*R/(P+R)
        "per_class": {  # доп. поля для анализа дисбаланса в отчёте
            "<class>": {"map50", "map50_95", "precision", "recall", "f1"}, ...
        },
        "num_images": int,   # размер сплита, на котором считались метрики
        "split": str,        # имя сплита ("test")
        "save_dir": str,     # каталог с весами и графиками
    }

Остальные модели должны возвращать метрики в этом же формате, чтобы
:mod:`src.evaluation.metrics` собирал их единообразно.

Специфика Ultralytics: ``model`` — обёртка ``ultralytics.YOLO`` (не
``torch.nn.Module``), а ``dataset`` здесь — **путь к ``data.yaml``**, а не HF
``Dataset``, как в остальном пайплайне подготовки. Так сделано потому, что
Ultralytics читает изображения и разметку сам по ``data.yaml`` и не принимает
готовый объект датасета.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from src.dataset.transforms import yolo_augmentation_args
from src.utils.utils import PROJECT_ROOT, get_logger

#: Число эпох для baseline-прогона (проверка пайплайна end-to-end).
#: Финальные 50 эпох из configs/default.yaml здесь НЕ используются намеренно.
BASELINE_EPOCHS = 20

#: Приведение имени оптимизатора из конфига к тому, что ожидает Ultralytics.
_OPTIMIZER_ALIASES = {
    "adam": "Adam",
    "adamw": "AdamW",
    "sgd": "SGD",
    "nadam": "NAdam",
    "radam": "RAdam",
    "rmsprop": "RMSProp",
    "auto": "auto",
}


def _normalize_optimizer(name: str) -> str:
    """Привести имя оптимизатора к принятому в Ultralytics написанию."""
    return _OPTIMIZER_ALIASES.get(str(name).lower(), name)


def _f1(precision: float, recall: float) -> float:
    """Гармоническое среднее точности и полноты (0 при нулевом знаменателе)."""
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0


def _extract_metrics(results, split: str, num_images: int, save_dir) -> dict:
    """Привести объект метрик Ultralytics к единому контракту проекта.

    ``results`` — объект ``DetMetrics`` из ``model.val``. Общие метрики берутся
    из усреднённых значений (``box.map50``, ``box.map``, ``box.mp``, ``box.mr``),
    per-class — из массивов, индексируемых ``box.ap_class_index`` (только для
    классов, реально присутствовавших в оценке).
    """
    box = results.box
    names = getattr(results, "names", {}) or {}

    metrics = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": _f1(float(box.mp), float(box.mr)),
        "split": split,
        "num_images": int(num_images),
        "save_dir": str(save_dir),
    }

    per_class: dict[str, dict] = {}
    for i, class_index in enumerate(box.ap_class_index):
        name = names.get(int(class_index), str(int(class_index)))
        precision = float(box.p[i])
        recall = float(box.r[i])
        per_class[name] = {
            "map50": float(box.ap50[i]),
            "map50_95": float(box.ap[i]),
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    metrics["per_class"] = per_class
    return metrics


def train(
    model,
    dataset,
    config: dict,
    *,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    split: str = "test",
    project: str | Path | None = None,
    name: str = "yolov8_baseline",
    logger=None,
    **train_overrides,
) -> dict:
    """Обучить YOLOv8 и вернуть метрики на указанном сплите (по умолчанию test).

    Параметры
    ---------
    model:
        Объект ``ultralytics.YOLO`` из :func:`src.models.yolo.build_model`.
    dataset:
        Путь к ``data.yaml`` Ultralytics (см. модульный докстринг о выборе).
    config:
        Конфигурация эксперимента; используются секции ``training`` и
        ``augmentation`` (аугментации берутся через ``yolo_augmentation_args``,
        логика не дублируется).
    epochs:
        Число эпох. Если ``None`` — используется :data:`BASELINE_EPOCHS` (20)
        для baseline-прогона; финальные 50 из конфига намеренно не берутся,
        чтобы не менять ``configs/default.yaml`` глобально.
    batch:
        Размер батча. Если ``None`` — берётся из ``training.batch_size``. При
        нехватке видеопамяти (OOM) обучение автоматически повторяется с
        ``batch=8``.
    split:
        Сплит для итоговой оценки. По умолчанию ``"test"`` — честная отложенная
        оценка (val участвует в валидации во время обучения). Тот же test будет
        использоваться остальными 4 моделями для сопоставимости.
    project, name:
        Каталог результатов Ultralytics: ``{project}/{name}``. По умолчанию
        ``results/logs/yolov8_baseline`` в корне проекта.
    train_overrides:
        Любые дополнительные аргументы, пробрасываемые в ``model.train``.

    Возвращает
    ----------
    Словарь метрик в едином контракте проекта (см. модульный докстринг).
    """
    logger = logger or get_logger()
    training = config.get("training", {})
    data_path = str(dataset)

    if project is None:
        project = PROJECT_ROOT / "results" / "logs"
    project = str(project)

    epochs = epochs if epochs is not None else BASELINE_EPOCHS
    batch = batch if batch is not None else training.get("batch_size", 16)

    train_args = dict(
        data=data_path,
        epochs=epochs,
        imgsz=training.get("image_size", 640),
        batch=batch,
        optimizer=_normalize_optimizer(training.get("optimizer", "auto")),
        lr0=training.get("learning_rate", 0.01),
        seed=training.get("seed", 42),
        project=project,
        name=name,
        exist_ok=True,
    )
    train_args.update(yolo_augmentation_args(config))
    train_args.update(train_overrides)

    logger.info(
        "YOLOv8: старт обучения — epochs=%d, batch=%d, imgsz=%d, optimizer=%s",
        epochs, train_args["batch"], train_args["imgsz"], train_args["optimizer"],
    )

    try:
        model.train(**train_args)
    except Exception as error:  # OOM: повтор с уменьшенным батчем
        message = str(error).lower()
        is_oom = "out of memory" in message or ("cuda" in message and "memory" in message)
        if is_oom and train_args["batch"] > 8:
            logger.warning(
                "Нехватка видеопамяти (OOM). Повтор обучения с batch=8. "
                "Исходная ошибка: %s", error,
            )
            _free_cuda()
            train_args["batch"] = 8
            model.train(**train_args)
        else:
            raise

    logger.info("Обучение завершено. Оценка на сплите '%s'...", split)
    # Оцениваем ЛУЧШИЕ веса (best.pt), а не последние в памяти, — это честная
    # итоговая оценка. Если best.pt не найден, используем текущую модель.
    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        from ultralytics import YOLO

        eval_model = YOLO(str(best_weights))
        logger.info("Оценка по лучшим весам: %s", best_weights)
    else:
        eval_model = model

    val_results = eval_model.val(
        data=data_path,
        split=split,
        imgsz=train_args["imgsz"],
        batch=train_args["batch"],
        project=project,
        name=f"{name}_{split}",
        exist_ok=True,
        plots=True,
    )

    save_dir = getattr(val_results, "save_dir", Path(project) / f"{name}_{split}")
    num_images = _count_split_images(data_path, split)
    metrics = _extract_metrics(val_results, split, num_images, save_dir)

    metrics_path = Path(save_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    logger.info("Метрики сохранены: %s", metrics_path)

    return metrics


def _free_cuda() -> None:
    """Освободить кеш CUDA перед повторной попыткой обучения (best-effort)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - вспомогательная очистка
        pass


# ---------------------------------------------------------------------------
# torchvision-детекторы (Faster R-CNN, SSD) — общий цикл обучения и оценки.
# ---------------------------------------------------------------------------

#: Целевое число эпох (совпадает с YOLOv8-baseline для честного сравнения).
TORCHVISION_EPOCHS = 20

#: Минимальный batch, ниже которого при OOM опускаться не имеет смысла.
_MIN_BATCH = 1


def _seed_torch(seed: int) -> None:
    """Зафиксировать генераторы (torch/numpy/random) для воспроизводимости."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_schedule(start: int) -> list[int]:
    """Последовательность batch для отката при OOM: start, /2, ... , 1."""
    schedule, value = [], int(start)
    while value >= _MIN_BATCH:
        schedule.append(value)
        if value == _MIN_BATCH:
            break
        value = max(_MIN_BATCH, value // 2)
    return schedule


def train_torchvision_detector(
    model,
    train_loader,
    eval_loader,
    config: dict,
    *,
    model_name: str,
    class_names: Sequence[str],
    epochs: int = TORCHVISION_EPOCHS,
    device: Optional[str] = None,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    project: str | Path | None = None,
    label_offset: int = 1,
    logger=None,
) -> dict:
    """Общий цикл обучения torchvision detection моделей (Faster R-CNN и SSD).

    Одна функция на обе модели (без дублирования). Обучает на ``train_loader``,
    затем оценивает на ``eval_loader`` (передавать **test**-сплит для итоговой
    честной оценки, как у YOLOv8) через :func:`evaluate_coco_detector`.

    Оптимизатор — **SGD** (lr≈0.005, momentum 0.9, weight_decay 5e-4). Для
    архитектур Faster R-CNN / SSD это общепринятая практика, дающая устойчивую
    сходимость; ``optimizer: adam`` из ``configs/default.yaml`` относится к
    YOLOv8-baseline и намеренно НЕ используется здесь (в §3.5 у каждой модели
    своя строка гиперпараметров). ``epochs`` и ``seed`` — как у baseline.

    Логирует loss по эпохам (консоль + ``results.csv``, аналогично YOLOv8),
    сохраняет веса ``<model_name>.pth`` и ``metrics.json`` в
    ``results/logs/<model_name>/``.
    """
    import torch

    logger = logger or get_logger()
    training_cfg = config.get("training", {})
    seed = training_cfg.get("seed", 42)
    _seed_torch(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if project is None:
        project = PROJECT_ROOT / "results" / "logs"
    out_dir = Path(project) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(epochs * 0.7)), gamma=0.1
    )

    csv_path = out_dir / "results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = None

    logger.info(
        "%s: старт обучения — epochs=%d, device=%s, optimizer=SGD(lr=%.4g, "
        "momentum=%.2g, wd=%.4g)",
        model_name, epochs, device, lr, momentum, weight_decay,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        components: dict[str, float] = defaultdict(float)
        num_batches = 0

        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            for key, value in loss_dict.items():
                components[key] += float(value.item())
            num_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, num_batches)
        avg_components = {k: v / max(1, num_batches) for k, v in components.items()}
        elapsed = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            **{k: round(v, 6) for k, v in avg_components.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(elapsed, 1),
        }
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()

        logger.info(
            "%s | эпоха %2d/%d | loss=%.4f | lr=%.2g | %.0f c",
            model_name, epoch, epochs, avg_loss,
            optimizer.param_groups[0]["lr"], elapsed,
        )

    csv_file.close()

    weights_path = out_dir / f"{model_name}.pth"
    torch.save(model.state_dict(), weights_path)
    logger.info("%s: веса сохранены -> %s", model_name, weights_path)

    logger.info("%s: оценка на test через COCOeval...", model_name)
    from src.evaluation.metrics import evaluate_coco_detector

    metrics = evaluate_coco_detector(
        model, eval_loader, device, class_names,
        label_offset=label_offset, logger=logger,
    )
    metrics.update(
        {"save_dir": str(out_dir), "weights": str(weights_path),
         "epochs": epochs, "model": model_name}
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    logger.info("%s: метрики сохранены -> %s", model_name, out_dir / "metrics.json")
    return metrics


def run_torchvision_training(
    model_builder: Callable[[int], object],
    num_classes: int,
    train_ann: str | Path,
    eval_ann: str | Path,
    data_root: str | Path,
    config: dict,
    *,
    model_name: str,
    class_names: Sequence[str],
    batch_size: int = 4,
    epochs: int = TORCHVISION_EPOCHS,
    label_offset: int = 1,
    num_workers: int = 2,
    subset_size: Optional[int] = None,
    project: str | Path | None = None,
    logger=None,
    **train_kwargs,
) -> dict:
    """Собрать loaders, модель и обучить с авто-откатом batch при OOM.

    Faster R-CNN / SSD с ResNet/VGG backbone тяжелее YOLOv8 по памяти, на
    Tesla T4 (14 ГБ) обычно нужен batch 4–8. При ``CUDA out of memory`` batch
    последовательно уменьшается (``batch, /2, ... , 1``), модель и loaders
    пересобираются заново.

    ``model_builder(num_classes)`` — фабрика модели
    (``src.models.faster_rcnn.build_model`` / ``src.models.ssd.build_model``).
    ``eval_ann`` — аннотации сплита для итоговой оценки (передавать test).
    """
    from src.dataset.coco_dataset import build_loader

    logger = logger or get_logger()

    last_error: Optional[Exception] = None
    for attempt_batch in _batch_schedule(batch_size):
        try:
            train_loader = build_loader(
                train_ann, data_root, batch_size=attempt_batch, shuffle=True,
                num_workers=num_workers, label_offset=label_offset,
                subset_size=subset_size,
            )
            eval_loader = build_loader(
                eval_ann, data_root, batch_size=1, shuffle=False,
                num_workers=num_workers, label_offset=label_offset,
                subset_size=subset_size,
            )
            model = model_builder(num_classes)
            logger.info("%s: попытка обучения с batch=%d", model_name, attempt_batch)
            return train_torchvision_detector(
                model, train_loader, eval_loader, config,
                model_name=model_name, class_names=class_names, epochs=epochs,
                label_offset=label_offset, project=project, logger=logger,
                **train_kwargs,
            )
        except RuntimeError as error:
            last_error = error
            if "out of memory" in str(error).lower() and attempt_batch > _MIN_BATCH:
                logger.warning(
                    "%s: OOM при batch=%d — уменьшаю batch и пробую снова. %s",
                    model_name, attempt_batch, error,
                )
                _free_cuda()
                continue
            raise

    raise RuntimeError(
        f"{model_name}: не удалось обучить даже при batch={_MIN_BATCH}. "
        f"Исходная ошибка: {last_error}"
    )


def _count_split_images(data_yaml: str | Path, split: str) -> int:
    """Оценить число изображений в сплите по путям из ``data.yaml``."""
    import yaml

    try:
        with open(data_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        rel = data.get(split)
        if rel is None:
            return 0
        root = data.get("path")
        images_dir = Path(root) / rel if root else Path(data_yaml).parent / rel
        if not images_dir.is_absolute():
            images_dir = Path(data_yaml).parent / images_dir
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sum(1 for p in images_dir.glob("*") if p.suffix.lower() in exts)
    except Exception:  # pragma: no cover - только для сводки
        return 0
