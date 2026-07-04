# Модель YOLOv8 - наша базовая модель (baseline).
#
# YOLOv8 берём из библиотеки Ultralytics. Это не обычная сеть на PyTorch, а
# готовый объект YOLO со своими методами: .train (обучение), .val (проверка),
# .predict (предсказание). Поэтому build_model возвращает именно такой объект.
#
# Начинаем с предобученных весов yolov8n.pt - это самая маленькая и быстрая
# модель семейства, удобно для отладки.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

# Веса по умолчанию (самая лёгкая модель семейства).
DEFAULT_WEIGHTS = "yolov8n.pt"


def _read_nc(data_yaml: str | Path) -> int:
    """Прочитать число классов nc из файла data.yaml."""
    with open(data_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data.get("nc") is not None:
        return int(data["nc"])
    # Если nc нет, считаем классы по длине списка имён.
    return len(data.get("names", []))


def build_model(
    num_classes: int,
    config: dict | None = None,
    weights: Optional[str] = None,
    data_yaml: str | Path | None = None,
):
    """Создать модель YOLOv8, готовую к обучению.

    num_classes - число классов одежды. Для YOLOv8 оно нужно только для
    проверки: реальное число классов Ultralytics сам берёт из data.yaml при
    обучении. Если передать data_yaml, функция сверит num_classes с числом
    классов в файле и выдаст понятную ошибку, если они не совпадают.

    config - настройки; из них может браться имя весов (model.weights).
    Остальные настройки обучения (размер картинки, batch и т.д.) передаются не
    сюда, а в функцию train, потому что у Ultralytics это аргументы .train.

    weights - имя или путь к весам, по умолчанию yolov8n.pt.
    data_yaml - необязательный путь к data.yaml для проверки числа классов.

    Возвращает объект YOLO, готовый к .train(...).
    """
    try:
        from ultralytics import YOLO
    except ImportError as error:  # библиотека может быть не установлена
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
