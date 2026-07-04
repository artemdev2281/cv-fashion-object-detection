# Подготовка датасета Fashionpedia к обучению.
#
# Здесь собран весь процесс по шагам: отобрать нужные классы -> почистить
# рамки -> разбить на train/val/test -> сохранить в форматах YOLO и COCO ->
# записать список классов -> напечатать сводку.
# Запускается через: python main.py --prepare-data
# Настройки берутся из configs/default.yaml.

from __future__ import annotations

from pathlib import Path

from src.dataset.convert import export_dataset
from src.dataset.dataset import build_splits
from src.utils.utils import PROJECT_ROOT, get_logger


def _log_summary(summary: dict, class_names, logger) -> None:
    """Напечатать таблицу: сколько картинок и объектов каждого класса в каждой части."""
    logger.info("Сводка подготовленных данных:")
    header = f"{'класс':<28}" + "".join(f"{s:>10}" for s in summary)
    logger.info(header)
    for new_id, name in enumerate(class_names):
        row = f"{new_id:2d} {name:<25}"
        for split in summary:
            row += f"{summary[split]['class_counts'][name]:>10}"
        logger.info(row)
    logger.info("-" * len(header))
    totals = f"{'изображений':<28}" + "".join(
        f"{summary[s]['images']:>10}" for s in summary
    )
    objects = f"{'объектов':<28}" + "".join(
        f"{summary[s]['objects']:>10}" for s in summary
    )
    logger.info(totals)
    logger.info(objects)


def prepare_dataset(config: dict, logger=None) -> dict:
    """Пройти все шаги подготовки данных и вернуть сводку.

    Все настройки берёт из config (разделы dataset, training, paths).
    """
    logger = logger or get_logger()

    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    selected = dataset_cfg.get("categories")
    if not selected:
        raise ValueError(
            "В configs/default.yaml не задан список dataset.categories "
            "(отобранные по EDA категории)."
        )
    if dataset_cfg.get("streaming"):
        logger.warning(
            "Потоковый режим (streaming) несовместим с подготовкой данных "
            "(нужен произвольный доступ и сохранение изображений); отключаю."
        )

    processed_dir = Path(paths_cfg.get("processed_data", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir

    seed = training_cfg.get("seed", 42)
    test_size = dataset_cfg.get("splits", {}).get("test_size", 0.1)
    subset_size = dataset_cfg.get("subset_size")
    formats = dataset_cfg.get("formats", ["yolo", "coco"])

    logger.info("Загрузка и фильтрация датасета (%d категорий)...", len(selected))
    try:
        dataset_dict, class_names, orig_to_new = build_splits(
            selected_names=selected,
            test_size=test_size,
            subset_size=subset_size,
            seed=seed,
        )
    except Exception as error:  # диагностическое сообщение для среды Kaggle
        raise RuntimeError(
            "Не удалось подготовить сплиты. Возможные причины: отсутствует "
            "интернет (датасет не скачан) или не хватает места на диске. "
            f"Исходная ошибка: {error}"
        ) from error

    logger.info(
        "Отобрано классов: %d. Размеры сплитов — train: %d, val: %d, test: %d",
        len(class_names),
        len(dataset_dict["train"]),
        len(dataset_dict["val"]),
        len(dataset_dict["test"]),
    )

    logger.info("Экспорт в форматы: %s -> %s", ", ".join(formats), processed_dir)
    try:
        summary = export_dataset(
            dataset_dict=dataset_dict,
            class_names=class_names,
            orig_to_new=orig_to_new,
            out_dir=processed_dir,
            formats=formats,
        )
    except OSError as error:
        raise OSError(
            "Ошибка записи подготовленных данных. Вероятно, недостаточно места "
            "на диске (лимит Kaggle /kaggle/working — 20 ГБ). Уменьшите объём "
            "через dataset.subset_size в конфигурации. "
            f"Исходная ошибка: {error}"
        ) from error

    _log_summary(summary, class_names, logger)
    logger.info("Подготовка данных завершена. Данные сохранены в %s", processed_dir)
    return summary
