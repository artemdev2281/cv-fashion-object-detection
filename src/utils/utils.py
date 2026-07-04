# Разные вспомогательные функции: чтение конфига, логи, рисование рамок
# и построение графиков для отчёта. Общие для всего проекта.

from __future__ import annotations

import logging
from pathlib import Path

import yaml

# Корневая папка проекта.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Конфиг по умолчанию.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Прочитать настройки из YAML-файла."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_logger(name: str = "fashion_detection", log_file: str | Path | None = None) -> logging.Logger:
    """Сделать логгер, который пишет в консоль (и при желании в файл)."""
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
    """Нарисовать рамки на картинке.

    image - картинка, boxes - список рамок [x_min, y_min, x_max, y_max].
    labels - номера классов (не обязательно), category_names - их названия,
    чтобы подписать рамки. ax - на каком графике рисовать (если None, создаём
    новый). Возвращает график с нарисованными рамками.
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


# ---------------------------------------------------------------------------
# Графики для отчёта. Берём логи из results/logs/, картинки кладём в
# results/plots/. Логи - это CSV/JSON, которые получились при обучении 5 моделей.
# ---------------------------------------------------------------------------

# Папки с результатами.
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"

# Список наших 5 моделей: (короткое имя, красивое имя, папка с логами).
_MODELS: list[tuple[str, str, str]] = [
    ("yolov8", "YOLOv8n", "yolov8_subset3000"),
    ("faster_rcnn", "Faster R-CNN", "faster_rcnn"),
    ("ssd", "SSD", "ssd"),
    ("efficientdet", "EfficientDet", "efficientdet"),
    ("detr", "DETR", "detr"),
]

# Свой цвет для каждой модели, чтобы на всех графиках они были одинаковыми.
_MODEL_COLORS: dict[str, str] = {
    "YOLOv8n": "#1f77b4",
    "Faster R-CNN": "#2ca02c",
    "SSD": "#ff7f0e",
    "EfficientDet": "#9467bd",
    "DETR": "#d62728",
}

# Сколько минут обучалась каждая модель (на видеокарте Tesla T4, 3000 картинок,
# 20 эпох). Нужно для графика "точность против скорости". Взято из логов.
_TRAIN_MINUTES: dict[str, float] = {
    "YOLOv8n": 12.0,
    "Faster R-CNN": 138.0,
    "SSD": 22.0,
    "EfficientDet": 44.0,
    "DETR": 342.0,
}


def _save_fig(fig, path: Path, logger=None) -> None:
    """Сохранить график в PNG и закрыть его."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info("График сохранён: %s", path)


def _load_loss_series(results_csv: Path):
    """Достать номера эпох и лосс из файла results.csv.

    У большинства моделей лосс лежит в колонке train_loss. А у YOLOv8 он
    разбит на несколько колонок (box_loss, cls_loss, dfl_loss), поэтому их
    складываем, чтобы получить общий лосс.
    """
    import pandas as pd

    frame = pd.read_csv(results_csv)
    frame.columns = [c.strip() for c in frame.columns]
    epochs = frame["epoch"].to_numpy() if "epoch" in frame.columns else range(1, len(frame) + 1)
    if "train_loss" in frame.columns:
        loss = frame["train_loss"].to_numpy()
    else:  # YOLOv8: складываем части лосса в один
        loss_cols = [c for c in frame.columns if c.startswith("train/") and c.endswith("loss")]
        loss = frame[loss_cols].sum(axis=1).to_numpy()
    return epochs, loss


def plot_loss_curves(logs_dir: Path = LOGS_DIR, plots_dir: Path = PLOTS_DIR, logger=None) -> None:
    """Нарисовать графики лосса по эпохам - по одному на каждую модель."""
    import matplotlib.pyplot as plt

    for key, name, subdir in _MODELS:
        results_csv = logs_dir / subdir / "results.csv"
        if not results_csv.exists():
            continue
        epochs, loss = _load_loss_series(results_csv)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, loss, marker="o", markersize=3, color=_MODEL_COLORS[name], linewidth=1.8)
        ax.set_title(f"Кривая обучающего лосса — {name}")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Суммарный обучающий лосс")
        ax.grid(True, alpha=0.3)
        _save_fig(fig, plots_dir / f"loss_{key}.png", logger)


def plot_per_class(logs_dir: Path = LOGS_DIR, plots_dir: Path = PLOTS_DIR, logger=None) -> None:
    """Нарисовать mAP@0.5 по каждому классу - свой столбчатый график на модель."""
    import matplotlib.pyplot as plt
    import pandas as pd

    for key, name, subdir in _MODELS:
        per_class_csv = logs_dir / subdir / "per_class.csv"
        if not per_class_csv.exists():
            continue
        frame = pd.read_csv(per_class_csv, index_col=0).sort_values("map50", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(frame.index.astype(str), frame["map50"], color=_MODEL_COLORS[name])
        ax.set_title(f"mAP@0.5 по классам — {name}")
        ax.set_xlabel("mAP@0.5")
        ax.set_xlim(0, 1)
        for y, value in enumerate(frame["map50"]):
            ax.text(value + 0.01, y, f"{value:.3f}", va="center", fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        _save_fig(fig, plots_dir / f"perclass_{key}.png", logger)


def plot_comparison(logs_dir: Path = LOGS_DIR, plots_dir: Path = PLOTS_DIR, logger=None) -> None:
    """Один общий столбчатый график: 5 метрик для всех 5 моделей рядом."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    comparison_csv = logs_dir / "final_comparison_5_models.csv"
    if not comparison_csv.exists():
        return
    frame = pd.read_csv(comparison_csv, index_col=0)
    metrics = ["map50", "map50_95", "precision", "recall", "f1"]
    metric_labels = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
    models = [name for _, name, _ in _MODELS if name in frame.index]

    x = np.arange(len(metrics))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, name in enumerate(models):
        values = [frame.loc[name, m] for m in metrics]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width,
               label=name, color=_MODEL_COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение 5 моделей детектирования (test, train=3000, 20 эпох)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, plots_dir / "comparison_map_pr.png", logger)


def plot_accuracy_vs_speed(logs_dir: Path = LOGS_DIR, plots_dir: Path = PLOTS_DIR, logger=None) -> None:
    """График "точность против скорости": mAP@0.5 и время обучения."""
    import matplotlib.pyplot as plt
    import pandas as pd

    comparison_csv = logs_dir / "final_comparison_5_models.csv"
    if not comparison_csv.exists():
        return
    frame = pd.read_csv(comparison_csv, index_col=0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, name, _ in _MODELS:
        if name not in frame.index:
            continue
        minutes = _TRAIN_MINUTES[name]
        map50 = frame.loc[name, "map50"]
        ax.scatter(minutes, map50, s=120, color=_MODEL_COLORS[name], zorder=3)
        ax.annotate(f"  {name}", (minutes, map50), fontsize=9, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Время обучения, мин (лог. шкала, Tesla T4)")
    ax.set_ylabel("mAP@0.5 (test)")
    ax.set_title("Точность против скорости обучения")
    ax.grid(True, which="both", alpha=0.3)
    _save_fig(fig, plots_dir / "accuracy_vs_speed.png", logger)


def plot_hp_experiments(logs_dir: Path = LOGS_DIR, plots_dir: Path = PLOTS_DIR, logger=None) -> None:
    """Столбчатые графики: как менялось качество при подборе гиперпараметров
    для YOLO и SSD."""
    import matplotlib.pyplot as plt
    import pandas as pd

    for model_key, title in (("yolo", "YOLOv8"), ("ssd", "SSD")):
        hp_csv = logs_dir / "hp" / f"hp_{model_key}.csv"
        if not hp_csv.exists():
            continue
        frame = pd.read_csv(hp_csv, index_col=0).sort_values("map50", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(frame.index.astype(str), frame["map50"], color="#4c78a8")
        ax.set_title(f"Подбор гиперпараметров — {title} (mAP@0.5, test)")
        ax.set_ylabel("mAP@0.5")
        ax.set_ylim(0, max(0.6, float(frame["map50"].max()) * 1.15))
        for i, value in enumerate(frame["map50"]):
            ax.text(i, value + 0.005, f"{value:.3f}", ha="center", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        _save_fig(fig, plots_dir / f"hp_{model_key}.png", logger)


def copy_yolo_ready_figures(logs_dir: Path = LOGS_DIR, plots_dir: Path = PLOTS_DIR, logger=None) -> None:
    """Скопировать готовые графики YOLOv8 (их рисует сам Ultralytics) в results/plots/."""

    import shutil

    plots_dir.mkdir(parents=True, exist_ok=True)
    src = logs_dir / "yolov8_subset3000"
    ready = {
        "results.png": "yolov8_training_curves.png",
        "BoxPR_curve.png": "yolov8_pr_curve.png",
        "confusion_matrix_normalized.png": "yolov8_confusion_matrix.png",
        "val_batch0_pred.jpg": "yolov8_detection_example.jpg",
    }
    for source_name, target_name in ready.items():
        source = src / source_name
        if source.exists():
            shutil.copy(source, plots_dir / target_name)
            if logger:
                logger.info("Скопирован готовый график YOLOv8: %s", plots_dir / target_name)


def generate_report_figures(logs_dir: Path = LOGS_DIR, plots_dir: Path = PLOTS_DIR, logger=None) -> None:
    """Построить сразу все графики для отчёта из логов в results/logs/.

    Строит всё: 5 графиков лосса, 5 графиков по классам, общее сравнение,
    точность против скорости, 2 графика по подбору гиперпараметров, плюс
    копирует готовые графики YOLOv8. Всё сохраняется как PNG в results/plots/.
    """
    logger = logger or get_logger()
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Генерация графиков отчёта: %s -> %s", logs_dir, plots_dir)
    plot_loss_curves(logs_dir, plots_dir, logger)
    plot_per_class(logs_dir, plots_dir, logger)
    plot_comparison(logs_dir, plots_dir, logger)
    plot_accuracy_vs_speed(logs_dir, plots_dir, logger)
    plot_hp_experiments(logs_dir, plots_dir, logger)
    copy_yolo_ready_figures(logs_dir, plots_dir, logger)
    logger.info("Готово. Графики в %s", plots_dir)


if __name__ == "__main__":  # python -m src.utils.utils
    generate_report_figures()
