"""Загрузка и предобработка набора данных Fashionpedia.

Модуль реализует загрузку датасета ``detection-datasets/fashionpedia`` с
платформы Hugging Face и его базовую предобработку: фильтрацию по выбранным
категориям, стратифицированное разбиение обучающей выборки на train/test
(официальная валидационная выборка используется как val) и фиксацию seed
для воспроизводимости.

Модуль рассчитан на запуск в Kaggle Notebooks с включённым интернетом.
Для среды с ограниченным дисковым пространством предусмотрены загрузка
подвыборки (``subset_size``) и потоковый режим (``streaming``).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

#: Имя датасета на Hugging Face.
HF_DATASET_NAME = "detection-datasets/fashionpedia"

#: Каталог для размещения исходных данных (data/raw в корне проекта).
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def set_seed(seed: int = 42) -> None:
    """Зафиксировать генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_fashionpedia(
    streaming: bool = False,
    subset_size: Optional[int] = None,
    cache_dir: os.PathLike | str = DEFAULT_RAW_DIR,
    seed: int = 42,
):
    """Загрузить датасет Fashionpedia с Hugging Face.

    Параметры
    ---------
    streaming:
        Потоковая загрузка без полного скачивания на диск. Полезно при
        ограниченном дисковом пространстве; в этом режиме недоступно
        стратифицированное разбиение (см. :func:`prepare_data`).
    subset_size:
        Если задано, из каждого split берутся первые ``subset_size`` примеров
        (для отладки и работы с ограниченным диском).
    cache_dir:
        Каталог кеша/загрузки датасета. По умолчанию ``data/raw``.
    seed:
        Базовый seed для воспроизводимости.

    Возвращает
    ----------
    ``DatasetDict`` (или словарь ``IterableDataset`` в режиме streaming) со
    split-ами ``train`` и ``val``.
    """
    from datasets import load_dataset

    set_seed(seed)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if streaming:
        dataset = {
            split: load_dataset(
                HF_DATASET_NAME, split=split, streaming=True
            )
            for split in ("train", "val")
        }
        if subset_size is not None:
            dataset = {
                split: ds.take(subset_size) for split, ds in dataset.items()
            }
        return dataset

    dataset = load_dataset(HF_DATASET_NAME, cache_dir=str(cache_dir))
    if subset_size is not None:
        dataset = {
            split: ds.select(range(min(subset_size, len(ds))))
            for split, ds in dataset.items()
        }
    return dataset


def get_category_names(dataset) -> list[str]:
    """Извлечь список наименований категорий из признаков датасета.

    Категориям в датасете соответствуют числовые идентификаторы (0–45);
    функция возвращает их строковые наименования в порядке идентификаторов.
    """
    category_feature = dataset.features["objects"].feature["category"]
    names = getattr(category_feature, "names", None)
    if names is None:
        raise ValueError(
            "Не удалось определить наименования категорий из признаков датасета."
        )
    return list(names)


def _resolve_category_ids(
    categories: Sequence[str | int],
    category_names: Sequence[str],
) -> set[int]:
    """Преобразовать имена/индексы выбранных категорий в множество id."""
    name_to_id = {name: idx for idx, name in enumerate(category_names)}
    resolved: set[int] = set()
    for category in categories:
        if isinstance(category, int):
            resolved.add(category)
        elif category in name_to_id:
            resolved.add(name_to_id[category])
        else:
            raise KeyError(f"Неизвестная категория: {category!r}")
    return resolved


def filter_by_categories(dataset, category_ids: set[int]):
    """Оставить в каждом примере только объекты выбранных категорий.

    Изображения, на которых после фильтрации не осталось объектов, удаляются.
    """

    def _filter_objects(example):
        objects = example["objects"]
        keep = [i for i, c in enumerate(objects["category"]) if c in category_ids]
        example["objects"] = {
            key: [values[i] for i in keep] for key, values in objects.items()
        }
        return example

    dataset = dataset.map(_filter_objects)
    dataset = dataset.filter(lambda ex: len(ex["objects"]["category"]) > 0)
    return dataset


def _add_primary_category(dataset, num_categories: int):
    """Добавить столбец ``primary_category`` — преобладающую категорию изображения.

    Категория используется как метка для стратифицированного разбиения.
    """
    from datasets import ClassLabel

    def _primary(example):
        cats = example["objects"]["category"]
        example["primary_category"] = max(set(cats), key=cats.count)
        return example

    dataset = dataset.map(_primary)
    dataset = dataset.cast_column(
        "primary_category", ClassLabel(num_classes=num_categories)
    )
    return dataset


def prepare_data(
    categories: Optional[Sequence[str | int]] = None,
    test_size: float = 0.1,
    subset_size: Optional[int] = None,
    streaming: bool = False,
    seed: int = 42,
    cache_dir: os.PathLike | str = DEFAULT_RAW_DIR,
):
    """Загрузить и предобработать Fashionpedia.

    Выполняет полный цикл подготовки данных:

    1. загрузка датасета с Hugging Face;
    2. (опционально) фильтрация по выбранным категориям;
    3. стратифицированное разбиение train на train/test (val — официальный).

    Параметры
    ---------
    categories:
        Имена или идентификаторы категорий, которые следует оставить.
        ``None`` — использовать все категории.
    test_size:
        Доля обучающей выборки, выделяемая под test.
    subset_size, streaming, seed, cache_dir:
        См. :func:`load_fashionpedia`.

    Возвращает
    ----------
    ``DatasetDict`` со split-ами ``train``, ``test``, ``val`` (в потоковом
    режиме разбиение train не выполняется и возвращаются ``train``/``val``).
    """
    set_seed(seed)
    dataset = load_fashionpedia(
        streaming=streaming,
        subset_size=subset_size,
        cache_dir=cache_dir,
        seed=seed,
    )

    category_names = get_category_names(dataset["train"])
    num_categories = len(category_names)

    if categories is not None:
        category_ids = _resolve_category_ids(categories, category_names)
        dataset = {
            split: filter_by_categories(ds, category_ids)
            for split, ds in dataset.items()
        }

    if streaming:
        # train_test_split недоступен для потоковых датасетов.
        return dataset

    from datasets import DatasetDict

    train_with_label = _add_primary_category(dataset["train"], num_categories)
    split = train_with_label.train_test_split(
        test_size=test_size,
        stratify_by_column="primary_category",
        seed=seed,
    )
    split = split.remove_columns("primary_category")

    return DatasetDict(
        train=split["train"],
        test=split["test"],
        val=dataset["val"],
    )
