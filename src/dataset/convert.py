# Сохранение подготовленного датасета в двух форматах: YOLO и COCO.
#
# Картинки сохраняем только один раз в папку images/{split}/, а оба формата на
# них ссылаются. Так картинки не дублируются и не занимают лишнее место (важно
# из-за лимита 20 ГБ в Kaggle). Разметка YOLO лежит рядом в labels/{split}/,
# а разметка COCO - в отдельных json-файлах.
#
# Рамки на входе в формате [x_min, y_min, x_max, y_max].
# YOLO хранит их как x_center y_center width height (числа от 0 до 1),
# COCO - как [x, y, width, height] в пикселях.

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import yaml


def save_class_mapping(
    class_names: Sequence[str],
    orig_to_new: dict[int, int],
    path: str | Path,
) -> None:
    """Сохранить в JSON, какой номер какому классу соответствует."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_to_orig = {new: orig for orig, new in orig_to_new.items()}
    mapping = {
        "num_classes": len(class_names),
        "names": {str(i): name for i, name in enumerate(class_names)},
        "new_to_original_id": {str(i): new_to_orig[i] for i in range(len(class_names))},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)


def write_data_yaml(
    out_dir: str | Path,
    class_names: Sequence[str],
    splits: Sequence[str] = ("train", "val", "test"),
) -> Path:
    """Создать файл data.yaml, который нужен YOLOv8 для обучения."""
    out_dir = Path(out_dir)
    data = {
        "path": str(out_dir.resolve()),
        "nc": len(class_names),
        "names": list(class_names),
    }
    for split in splits:
        data[split] = f"images/{split}"
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return yaml_path


def _yolo_line(new_id: int, box: Sequence[float], width: int, height: int) -> str:
    """Перевести одну рамку в строку формата YOLO (координаты от 0 до 1)."""
    x_min, y_min, x_max, y_max = box
    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return f"{new_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def export_dataset(
    dataset_dict,
    class_names: Sequence[str],
    orig_to_new: dict[int, int],
    out_dir: str | Path,
    formats: Sequence[str] = ("yolo", "coco"),
    image_quality: int = 95,
) -> dict:
    """Сохранить все части (train/val/test) в выбранные форматы и записать классы.

    Возвращает сводку: сколько картинок и объектов в каждой части и сколько
    объектов каждого класса (удобно проверить, что классы поделились ровно).
    """
    out_dir = Path(out_dir)
    formats = set(formats)
    images_root = out_dir / "images"
    labels_root = out_dir / "labels"

    summary: dict[str, dict] = {}

    for split, dataset in dataset_dict.items():
        images_dir = images_root / split
        images_dir.mkdir(parents=True, exist_ok=True)
        if "yolo" in formats:
            (labels_root / split).mkdir(parents=True, exist_ok=True)

        coco_images: list[dict] = []
        coco_annotations: list[dict] = []
        annotation_id = 1
        class_counts = [0] * len(class_names)
        num_objects = 0

        for example in dataset:
            image_id = example["image_id"]
            width, height = example["width"], example["height"]
            file_name = f"{image_id}.jpg"

            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(images_dir / file_name, format="JPEG", quality=image_quality)

            objects = example["objects"]
            num_objects += len(objects["category"])
            for category in objects["category"]:
                class_counts[category] += 1

            if "yolo" in formats:
                lines = [
                    _yolo_line(cat, box, width, height)
                    for cat, box in zip(objects["category"], objects["bbox"])
                ]
                label_path = labels_root / split / f"{image_id}.txt"
                label_path.write_text("\n".join(lines), encoding="utf-8")

            if "coco" in formats:
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": f"images/{split}/{file_name}",
                        "width": width,
                        "height": height,
                    }
                )
                for cat, box in zip(objects["category"], objects["bbox"]):
                    x_min, y_min, x_max, y_max = box
                    coco_annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": cat,
                            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                            "area": (x_max - x_min) * (y_max - y_min),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

        if "coco" in formats:
            coco = {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [
                    {"id": i, "name": name} for i, name in enumerate(class_names)
                ],
            }
            coco_dir = out_dir / "coco"
            coco_dir.mkdir(parents=True, exist_ok=True)
            with open(coco_dir / f"{split}.json", "w", encoding="utf-8") as handle:
                json.dump(coco, handle, ensure_ascii=False)

        summary[split] = {
            "images": len(dataset),
            "objects": num_objects,
            "class_counts": {
                class_names[i]: class_counts[i] for i in range(len(class_names))
            },
        }

    if "yolo" in formats:
        write_data_yaml(out_dir, class_names, splits=tuple(dataset_dict.keys()))

    save_class_mapping(class_names, orig_to_new, out_dir / "classes.json")

    return summary
