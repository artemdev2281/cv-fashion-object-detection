# Загрузка и подготовка датасета Fashionpedia.
#
# Скачиваем датасет detection-datasets/fashionpedia с Hugging Face и немного
# его обрабатываем: оставляем только нужные классы, делим train на train/test
# (официальный val используем как val) и фиксируем seed, чтобы результат был
# одинаковым при каждом запуске.
#
# Рассчитано на запуск в Kaggle с интернетом. Если мало места на диске, можно
# взять только часть данных (subset_size) или включить streaming.

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Имя датасета на Hugging Face.
HF_DATASET_NAME = "detection-datasets/fashionpedia"

# Куда скачиваем исходные данные (папка data/raw в проекте).
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Все 46 классов Fashionpedia по порядку (номер класса = позиция в списке).
# Используем как запасной вариант, если в самом датасете имена классов не нашлись.
FASHIONPEDIA_CATEGORIES = [
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit",
    "cape",
    "glasses",
    "hat",
    "headband, head covering, hair accessory",
    "tie",
    "glove",
    "watch",
    "belt",
    "leg warmer",
    "tights, stockings",
    "sock",
    "shoe",
    "bag, wallet",
    "scarf",
    "umbrella",
    "hood",
    "collar",
    "lapel",
    "epaulette",
    "sleeve",
    "pocket",
    "neckline",
    "buckle",
    "zipper",
    "applique",
    "bead",
    "bow",
    "flower",
    "fringe",
    "ribbon",
    "rivet",
    "ruffle",
    "sequin",
    "tassel",
]


def set_seed(seed: int = 42) -> None:
    """Зафиксировать случайность, чтобы результат был одинаковым при каждом запуске."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_fashionpedia(
    streaming: bool = False,
    subset_size: Optional[int] = None,
    cache_dir: os.PathLike | str = DEFAULT_RAW_DIR,
    seed: int = 42,
):
    """Скачать датасет Fashionpedia с Hugging Face.

    streaming - читать данные "на лету", не скачивая всё на диск (удобно,
    когда мало места, но тогда не получится делать разбиение по классам).
    subset_size - взять только первые N примеров (для отладки или экономии места).
    cache_dir - куда сохранять, по умолчанию data/raw.
    seed - для воспроизводимости.

    Возвращает датасет с частями train и val.
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
    """Получить список названий классов из датасета.

    В датасете классы хранятся как числа (0-45), а функция возвращает их
    названия в том же порядке. Поле objects в датасете может быть устроено
    по-разному, поэтому пробуем оба варианта. Если названий классов в датасете
    нет, берём наш список FASHIONPEDIA_CATEGORIES.
    """
    objects_feature = dataset.features["objects"]
    # Иногда поле обёрнуто в Sequence - разворачиваем до самой структуры.
    struct = getattr(objects_feature, "feature", objects_feature)
    category_feature = struct["category"]
    # category тоже может быть обёрнут - берём то, что внутри.
    inner = getattr(category_feature, "feature", category_feature)
    names = getattr(inner, "names", None) or getattr(category_feature, "names", None)
    if names is None:
        return list(FASHIONPEDIA_CATEGORIES)
    return list(names)


def resolve_selected_categories(
    dataset,
    selected_names: Sequence[str],
) -> tuple[dict[int, int], list[str]]:
    """Связать выбранные классы (по именам) со старыми номерами в датасете.

    Старые номера классов находим по их именам (а не берём как есть) - так
    надёжнее, если порядок классов вдруг поменяется. Выбранным классам даём
    новые номера по порядку: 0, 1, 2, ... в том порядке, как они идут в
    selected_names.

    dataset - любая часть датасета (нужна только чтобы прочитать имена классов).
    selected_names - список выбранных классов в нужном порядке.

    Возвращает orig_to_new (старый номер -> новый номер) и class_names
    (имена в порядке новых номеров).
    """
    all_names = get_category_names(dataset)
    name_to_orig = {name: idx for idx, name in enumerate(all_names)}

    orig_to_new: dict[int, int] = {}
    class_names: list[str] = []
    for new_id, name in enumerate(selected_names):
        if name not in name_to_orig:
            raise KeyError(
                f"Категория {name!r} отсутствует в датасете. "
                f"Проверьте configs/default.yaml (dataset.categories)."
            )
        orig_to_new[name_to_orig[name]] = new_id
        class_names.append(name)
    return orig_to_new, class_names


def filter_remap_clean(dataset, orig_to_new: dict[int, int]):
    """Оставить нужные классы, поменять их номера и почистить рамки.

    Для каждой картинки:
    - оставляем только объекты выбранных классов;
    - меняем старые номера классов на новые;
    - обрезаем рамки по краям картинки, а совсем плохие (нулевого или
      отрицательного размера) выбрасываем;
    - если после этого на картинке не осталось объектов, убираем и её.

    Рамки в формате [x_min, y_min, x_max, y_max].
    """

    def _process(example):
        width, height = example["width"], example["height"]
        objects = example["objects"]
        bbox_ids, categories, boxes, areas = [], [], [], []
        for index, orig_category in enumerate(objects["category"]):
            if orig_category not in orig_to_new:
                continue
            x_min, y_min, x_max, y_max = objects["bbox"][index]
            # Обрезаем рамку, чтобы она не вылезала за края картинки.
            x_min = max(0.0, min(float(x_min), width))
            y_min = max(0.0, min(float(y_min), height))
            x_max = max(0.0, min(float(x_max), width))
            y_max = max(0.0, min(float(y_max), height))
            # Пропускаем рамки, у которых нет нормальной ширины или высоты.
            if x_max <= x_min or y_max <= y_min:
                continue
            bbox_ids.append(objects["bbox_id"][index])
            categories.append(orig_to_new[orig_category])
            boxes.append([x_min, y_min, x_max, y_max])
            areas.append((x_max - x_min) * (y_max - y_min))
        example["objects"] = {
            "bbox_id": bbox_ids,
            "category": categories,
            "bbox": boxes,
            "area": areas,
        }
        return example

    dataset = dataset.map(_process)
    dataset = dataset.filter(lambda ex: len(ex["objects"]["category"]) > 0)
    return dataset


def _class_frequencies(dataset, num_classes: int) -> list[int]:
    """Посчитать, сколько раз встречается каждый класс."""
    counts = [0] * num_classes
    for objects in dataset["objects"]:
        for category in objects["category"]:
            counts[category] += 1
    return counts


def _add_stratify_label(dataset, num_classes: int):
    """Добавить каждой картинке метку strat_label для аккуратного разбиения.

    На одной картинке может быть сразу несколько классов, поэтому разбить
    поровну по всем сразу сложно. Как упрощение берём для картинки самый
    редкий из её классов - так редкие классы точно попадут и в train, и в test.
    """
    from datasets import ClassLabel

    frequencies = _class_frequencies(dataset, num_classes)

    def _label(example):
        present = set(example["objects"]["category"])
        example["strat_label"] = min(present, key=lambda c: frequencies[c])
        return example

    dataset = dataset.map(_label)
    dataset = dataset.cast_column(
        "strat_label", ClassLabel(num_classes=num_classes)
    )
    return dataset


def stratified_split(train_dataset, test_size: float, seed: int, num_classes: int):
    """Разбить train на train/test так, чтобы классы делились поровну.

    Делим по картинкам, поэтому одна картинка не может попасть сразу и в train,
    и в test. Seed фиксируем, чтобы разбиение было одинаковым при каждом запуске.

    Если поделить классы поровну не получается (например, на маленькой
    подвыборке какого-то класса слишком мало), просто делим случайно и пишем
    предупреждение.
    """
    labelled = _add_stratify_label(train_dataset, num_classes)
    try:
        split = labelled.train_test_split(
            test_size=test_size,
            stratify_by_column="strat_label",
            seed=seed,
        )
    except ValueError:
        import warnings

        warnings.warn(
            "Не получилось поделить классы поровну (какого-то класса слишком "
            "мало), поэтому делим случайно. Обычно бывает на маленькой "
            "подвыборке (subset_size).",
            RuntimeWarning,
        )
        split = labelled.train_test_split(test_size=test_size, seed=seed)
    return split.remove_columns("strat_label")


def build_splits(
    selected_names: Sequence[str],
    test_size: float = 0.1,
    subset_size: Optional[int] = None,
    seed: int = 42,
    cache_dir: os.PathLike | str = DEFAULT_RAW_DIR,
):
    """Собрать готовые части train/val/test.

    По шагам: скачиваем -> находим номера выбранных классов -> чистим каждую
    часть -> делим train на train/test (официальный val оставляем как val).

    Возвращает сам датасет (train, val, test), список классов и словарь
    старый номер -> новый номер.
    """
    from datasets import DatasetDict

    set_seed(seed)
    dataset = load_fashionpedia(
        streaming=False,
        subset_size=subset_size,
        cache_dir=cache_dir,
        seed=seed,
    )

    orig_to_new, class_names = resolve_selected_categories(
        dataset["train"], selected_names
    )
    num_classes = len(class_names)

    filtered = {
        split: filter_remap_clean(ds, orig_to_new)
        for split, ds in dataset.items()
    }

    split = stratified_split(
        filtered["train"], test_size=test_size, seed=seed, num_classes=num_classes
    )

    dataset_dict = DatasetDict(
        train=split["train"],
        val=filtered["val"],
        test=split["test"],
    )
    return dataset_dict, class_names, orig_to_new
