"""Вспомогательные функции: конфигурация, логирование, визуализация.

Содержит общие утилиты, используемые на всех этапах проекта и не зависящие
от конкретной модели детектирования.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

#: Корневой каталог проекта.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Конфигурация по умолчанию.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Загрузить YAML-конфигурацию эксперимента."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_logger(name: str = "fashion_detection", log_file: str | Path | None = None) -> logging.Logger:
    """Создать логгер с выводом в консоль и (опционально) в файл."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def draw_boxes(image, boxes, labels=None, category_names=None, ax=None):
    """Наложить ограничивающие рамки на изображение.

    Параметры
    ---------
    image:
        Изображение (``PIL.Image`` или массив ``numpy``).
    boxes:
        Последовательность рамок в формате ``[x_min, y_min, x_max, y_max]``.
    labels:
        Последовательность числовых меток категорий (необязательно).
    category_names:
        Список наименований категорий для подписи рамок (необязательно).
    ax:
        Объект ``matplotlib`` Axes; при ``None`` создаётся новый.

    Возвращает
    ----------
    Объект ``matplotlib`` Axes с нанесёнными рамками.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots(1, figsize=(8, 8))

    ax.imshow(image)
    ax.axis("off")

    for index, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        if labels is not None:
            label = labels[index]
            if category_names is not None:
                label = category_names[label]
            ax.text(
                x_min,
                y_min - 4,
                str(label),
                color="white",
                fontsize=9,
                bbox=dict(facecolor="red", alpha=0.7, pad=1),
            )

    return ax
