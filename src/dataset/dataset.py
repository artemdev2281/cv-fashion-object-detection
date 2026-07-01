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

#: Канонический список 46 категорий Fashionpedia (порядок соответствует
#: числовым идентификаторам). Используется как запасной вариант, если у поля
#: ``category`` в датасете отсутствуют встроенные имена классов (ClassLabel).
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
        стратифицированное разбиение (см. :func:`stratified_split`).
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

    Поле ``objects`` может быть представлено двумя способами: как структура из
    параллельных списков (``features['objects']['category']``) либо как
    ``Sequence``-обёртка над структурой (``features['objects'].feature[...]``);
    функция поддерживает оба варианта. Если у поля ``category`` нет встроенных
    имён классов, используется :data:`FASHIONPEDIA_CATEGORIES`.
    """
    objects_feature = dataset.features["objects"]
    # Sequence-обёртка над структурой -> разворачиваем до самой структуры.
    struct = getattr(objects_feature, "feature", objects_feature)
    category_feature = struct["category"]
    # category может быть Sequence(ClassLabel) либо ClassLabel напрямую.
    inner = getattr(category_feature, "feature", category_feature)
    names = getattr(inner, "names", None) or getattr(category_feature, "names", None)
    if names is None:
        return list(FASHIONPEDIA_CATEGORIES)
    return list(names)


def resolve_selected_categories(
    dataset,
    selected_names: Sequence[str],
) -> tuple[dict[int, int], list[str]]:
    """Сопоставить выбранные категории (по именам) с id датасета.

    Исходные ``category_id`` восстанавливаются программно из имён категорий
    датасета (а не принимаются на веру), что защищает от рассинхронизации
    порядка классов. Новые последовательные id (0..N-1) назначаются согласно
    порядку имён в ``selected_names``.

    Параметры
    ---------
    dataset:
        Любой split датасета (для чтения имён категорий).
    selected_names:
        Упорядоченный список имён отобранных категорий; индекс имени в списке
        становится новым id класса.

    Возвращает
    ----------
    Кортеж ``(orig_to_new, class_names)``, где ``orig_to_new`` отображает
    исходный id -> новый id, а ``class_names`` — имена в порядке новых id.
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
    """Отфильтровать, переиндексировать и очистить аннотации.

    Для каждого изображения:

    * оставляются только объекты отобранных категорий (по исходному id);
    * исходные id заменяются на новые последовательные id (remap);
    * очистка аннотаций — координаты рамки обрезаются по границам изображения,
      вырожденные рамки (нулевая/отрицательная площадь) отбрасываются;
    * изображения без валидных объектов после обработки исключаются.

    Рамки предполагаются в формате Pascal VOC ``[x_min, y_min, x_max, y_max]``.
    """

    def _process(example):
        width, height = example["width"], example["height"]
        objects = example["objects"]
        bbox_ids, categories, boxes, areas = [], [], [], []
        for index, orig_category in enumerate(objects["category"]):
            if orig_category not in orig_to_new:
                continue
            x_min, y_min, x_max, y_max = objects["bbox"][index]
            # Очистка: обрезка по границам изображения.
            x_min = max(0.0, min(float(x_min), width))
            y_min = max(0.0, min(float(y_min), height))
            x_max = max(0.0, min(float(x_max), width))
            y_max = max(0.0, min(float(y_max), height))
            # Отбраковка вырожденных рамок.
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
    """Подсчитать число экземпляров каждого класса (по новым id)."""
    counts = [0] * num_classes
    for objects in dataset["objects"]:
        for category in objects["category"]:
            counts[category] += 1
    return counts


def _add_stratify_label(dataset, num_classes: int):
    """Добавить столбец ``strat_label`` для стратифицированного разбиения.

    Изображение является мультиметочным (несколько классов одновременно), для
    которого строгая multilabel-стратификация нетривиальна. В качестве
    практической аппроксимации в роли метки стратификации используется
    **самый редкий из присутствующих на изображении классов** (по глобальной
    частоте в train). Такой выбор лучше сохраняет редкие классы при разбиении,
    чем метка по преобладающему классу.
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
    """Разбить train на train/test со стратификацией по классам.

    Разбиение выполняется на уровне изображений, поэтому одно изображение не
    может попасть в оба сплита (защита от утечки данных). ``random seed``
    фиксируется для воспроизводимости.

    Если стратификация невозможна (например, на подвыборке некоторый класс
    представлен слишком малым числом примеров), выполняется обычное случайное
    разбиение с тем же фиксированным seed, о чём выводится предупреждение.
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
            "Стратифицированное разбиение невозможно (недостаточно примеров "
            "некоторого класса); выполняется случайное разбиение с фиксированным "
            "seed. Обычно возникает при работе с подвыборкой (subset_size).",
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
    """Собрать финальные сплиты train/val/test с фильтрацией и очисткой.

    Порядок операций: загрузка -> сверка и разрешение категорий -> фильтрация,
    remap и очистка каждого split -> стратифицированное разбиение официального
    train на train/test (официальный val используется как validation).

    Возвращает
    ----------
    Кортеж ``(dataset_dict, class_names, orig_to_new)``, где ``dataset_dict`` —
    ``DatasetDict`` со split-ами ``train``, ``val``, ``test``.
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
