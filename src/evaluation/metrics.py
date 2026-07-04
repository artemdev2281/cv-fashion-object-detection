"""Модуль для расчёта метрик качества детекции.

Используем COCO-формат аннотаций и считаем mAP через pycocotools.
Также считаем precision и recall при конкретном пороге (IoU=0.5, score_thr=0.5),
чтобы они были похожи на те, что выводит YOLOv8.
Результат возвращаем в виде словаря с полями: map50, map50_95, precision, recall, f1, per_class.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence


def _xywh_to_xyxy(box) -> tuple[float, float, float, float]:
    """Переводит рамку из формата (x, y, w, h) в (x1, y1, x2, y2)."""
    x, y, w, h = box
    return x, y, x + w, y + h


def _iou(box_a, box_b) -> float:
    """Считает IoU двух прямоугольников (пересечение / объединение)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    """Считает F1-меру по precision и recall."""
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0


def _precision_recall(
    coco_gt,
    results: list,
    cat_ids: Sequence[int],
    iou_thr: float = 0.5,
    score_thr: float = 0.5,
    image_ids: Optional[set] = None,
):
    """Считает precision и recall (общие и по каждому классу) жадным сопоставлением.

    Для каждого изображения и класса предсказания сортируются по уверенности.
    Затем каждое предсказание пытаются сопоставить с истинной рамкой того же класса
    при IoU >= iou_thr. Одна истинная рамка может быть использована только один раз.
    Если задан image_ids, то считаем только по этим изображениям.
    Возвращает (общие precision, recall) и словарь per_class.
    """
    gt_by = defaultdict(list)
    for ann in coco_gt.dataset["annotations"]:
        if image_ids is not None and ann["image_id"] not in image_ids:
            continue
        gt_by[(ann["image_id"], ann["category_id"])].append(_xywh_to_xyxy(ann["bbox"]))

    det_by = defaultdict(list)
    for res in results:
        if res["score"] < score_thr:
            continue
        det_by[(res["image_id"], res["category_id"])].append(
            (res["score"], _xywh_to_xyxy(res["bbox"]))
        )

    tp = fp = fn = 0
    tpc = defaultdict(int)
    fpc = defaultdict(int)
    fnc = defaultdict(int)

    for key in set(gt_by) | set(det_by):
        _, cat = key
        gts = gt_by.get(key, [])
        dets = sorted(det_by.get(key, []), key=lambda item: -item[0])
        matched = [False] * len(gts)
        for _, det_box in dets:
            best_gt, best_iou = -1, iou_thr
            for gi, gt_box in enumerate(gts):
                if matched[gi]:
                    continue
                value = _iou(det_box, gt_box)
                if value >= best_iou:
                    best_iou, best_gt = value, gi
            if best_gt >= 0:
                matched[best_gt] = True
                tp += 1
                tpc[cat] += 1
            else:
                fp += 1
                fpc[cat] += 1
        missed = matched.count(False)
        fn += missed
        fnc[cat] += missed

    def _pr(t, f_pos, f_neg):
        precision = t / (t + f_pos) if (t + f_pos) > 0 else 0.0
        recall = t / (t + f_neg) if (t + f_neg) > 0 else 0.0
        return precision, recall

    overall = _pr(tp, fp, fn)
    per_class = {cat: _pr(tpc[cat], fpc[cat], fnc[cat]) for cat in cat_ids}
    return overall, per_class


def evaluate_coco_detector(
    model,
    data_loader,
    device,
    class_names: Sequence[str],
    label_offset: int = 1,
    score_thr: float = 0.5,
    iou_thr: float = 0.5,
    logger=None,
) -> dict:
    """Оценивает модель детекции на COCO-датасете и возвращает метрики.

    Аргументы:
        model — обученная модель (переводится в eval-режим внутри).
        data_loader — загрузчик данных, созданный из CocoDetectionDataset.
        device — устройство (cpu/cuda).
        class_names — имена классов в порядке id 0..N-1 (для per_class отчёта).
        label_offset — сдвиг меток, который был применён в датасете (обычно 1).
                      Здесь мы вычитаем его, чтобы вернуть исходные id.
        score_thr, iou_thr — пороги для расчёта precision/recall.
        logger — функция для логирования (опционально).
    """
    import torch
    from pycocotools.cocoeval import COCOeval

    logger = logger.info if logger else (lambda *a, **k: None)
    coco_gt = data_loader.dataset.coco
    cat_ids = sorted(coco_gt.getCatIds())
    # Берём только те изображения, которые реально проходят через даталоадер
    eval_image_ids = list(data_loader.dataset.ids)

    model.eval()
    model.to(device)
    results: list = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].cpu().tolist()
                scores = output["scores"].cpu().tolist()
                labels = output["labels"].cpu().tolist()
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label) - label_offset, # возвращаем исходный id
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )

    # Если модель ничего не предсказала, возвращаем нулевые метрики
    if not results: 
        empty = {name: {"map50": 0.0, "map50_95": 0.0, "precision": 0.0,
                        "recall": 0.0, "f1": 0.0} for name in class_names}
        return {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "per_class": empty, "num_detections": 0}

    # Формируем COCO-формат предсказаний и запускаем оценку
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.catIds = cat_ids
    coco_eval.params.imgIds = eval_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50_95 = float(coco_eval.stats[0])
    map50 = float(coco_eval.stats[1])

    # per-class AP из массива precision
    precisions = coco_eval.eval["precision"]
    per_class_ap: dict[int, tuple[float, float]] = {}
    for k, cat in enumerate(cat_ids):
        pr = precisions[:, :, k, 0, -1]
        valid = pr[pr > -1]
        ap = float(valid.mean()) if valid.size else 0.0
        pr50 = precisions[0, :, k, 0, -1]
        valid50 = pr50[pr50 > -1]
        ap50 = float(valid50.mean()) if valid50.size else 0.0
        per_class_ap[cat] = (ap50, ap)

    # Считаем precision и recall в рабочей точке (IoU=0.5, score_thr)
    (precision, recall), per_class_pr = _precision_recall(
        coco_gt, results, cat_ids, iou_thr=iou_thr, score_thr=score_thr,
        image_ids=set(eval_image_ids),
    )

    # Собираем per_class словарь с именами классов
    cat_to_name = {cat: coco_gt.loadCats(cat)[0]["name"] for cat in cat_ids}
    per_class: dict[str, dict] = {}
    for cat in cat_ids:
        ap50, ap = per_class_ap[cat]
        p, r = per_class_pr[cat]
        per_class[cat_to_name.get(cat, str(cat))] = {
            "map50": ap50,
            "map50_95": ap,
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
        }

    metrics = {
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "per_class": per_class,
        "num_detections": len(results),
    }
    logger("Оценка COCO: mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f F1=%.4f",
           map50, map50_95, precision, recall, metrics["f1"])
    return metrics


def evaluate(model, dataset, config: dict | None = None) -> dict:
    """Обёртка для обратной совместимости (не используется).

    Для оценки используйте evaluate_coco_detector.
    """
    raise NotImplementedError(
        "Используйте evaluate_coco_detector(model, data_loader, device, class_names)."
    )
