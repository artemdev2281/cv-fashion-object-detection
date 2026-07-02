"""Модель YOLOv8 (одноэтапный детектор семейства YOLO).

Модуль инкапсулирует создание модели YOLOv8 на базе библиотеки Ultralytics.
YOLOv8 — это не классический ``torch.nn.Module`` с ручным ``forward``, а
высокоуровневая обёртка ``ultralytics.YOLO`` с собственным API обучения
(``.train``), валидации (``.val``) и инференса (``.predict``). Поэтому
:func:`build_model` возвращает именно объект ``YOLO``, готовый к передаче в
:func:`src.training.train.train`, а не низкоуровневую сеть.

Baseline стартует с предобученных весов ``yolov8n.pt`` (наименьшая модель
семейства) — приоритет скорости обучения при отладке пайплайна.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

#: Предобученные веса по умолчанию (baseline — самая лёгкая модель семейства).
DEFAULT_WEIGHTS = "yolov8n.pt"


def _read_nc(data_yaml: str | Path) -> int:
    """Прочитать число классов ``nc`` из ``data.yaml`` Ultralytics."""
    with open(data_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data.get("nc") is not None:
        return int(data["nc"])
    # nc может отсутствовать явно — тогда берётся длина names.
    return len(data.get("names", []))


def build_model(
    num_classes: int,
    config: dict | None = None,
    weights: Optional[str] = None,
    data_yaml: str | Path | None = None,
):
    """Создать модель YOLOv8, готовую к обучению.

    Параметры
    ---------
    num_classes:
        Число категорий одежды. Для YOLOv8 это значение **информационное**:
        реальное число классов Ultralytics берёт из ``nc`` в ``data.yaml`` на
        этапе ``.train``, а в конструктор ``YOLO`` оно не передаётся. Если задан
        ``data_yaml``, функция сверяет ``num_classes`` с ``nc`` и кидает
        понятную ошибку при несовпадении (защита от рассинхронизации данных и
        конфигурации).
    config:
        Конфигурация эксперимента (``configs/default.yaml``). Из неё может быть
        взято имя весов (``model.weights``), если не задан аргумент ``weights``.
        Прочие гиперпараметры обучения (image_size, batch_size, optimizer …)
        передаются не здесь, а в :func:`src.training.train.train`, поскольку у
        Ultralytics они являются аргументами ``.train``, а не конструктора.
    weights:
        Имя/путь предобученных весов или ``.yaml``-архитектуры. По умолчанию
        :data:`DEFAULT_WEIGHTS`.
    data_yaml:
        Необязательный путь к ``data.yaml`` для сверки ``num_classes`` с ``nc``.

    Возвращает
    ----------
    Объект ``ultralytics.YOLO`` с загруженными весами, готовый к ``.train(...)``.
    """
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - зависит от среды выполнения
        raise ImportError(
            "Для YOLOv8 требуется пакет ultralytics. "
            "Установите зависимости: pip install -r requirements.txt"
        ) from error

    if weights is None:
        weights = (config or {}).get("model", {}).get("weights", DEFAULT_WEIGHTS)

    if data_yaml is not None:
        nc = _read_nc(data_yaml)
        if nc != num_classes:
            raise ValueError(
                f"Несовпадение числа классов: num_classes={num_classes}, "
                f"но в {data_yaml} nc={nc}. Проверьте согласованность "
                f"подготовленных данных (data/processed) и конфигурации."
            )

    return YOLO(weights)
