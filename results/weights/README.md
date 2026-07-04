# Веса обученных моделей

Файлы весов (`*.pt`, `*.pth`) в git **не хранятся** (см. `.gitignore`) из-за большого
размера (Faster R-CNN и DETR — по ~159 МБ, что превышает лимит GitHub 100 МБ/файл).
Они лежат локально в этой папке и в выводе соответствующих Kaggle-ноутбуков.

## Финальные модели (train = подвыборка 3000, 20 эпох, seed 42)

| Файл | Модель | Размер |
|------|--------|--------|
| `yolov8n.pt` | YOLOv8n | 6 МБ |
| `faster_rcnn.pth` | Faster R-CNN R50-FPN | 159 МБ |
| `ssd.pth` | SSD300-VGG16 | 96 МБ |
| `efficientdet.pth` | EfficientDet-d0 | 16 МБ |
| `detr.pth` | DETR-R50 | 159 МБ |

## Гиперпараметрические эксперименты (`hp/`)

YOLO: `yolo_adam_lr001`, `yolo_adam_lr005`, `yolo_adam_mosaic`, `yolo_sgd_lr01` (по 6 МБ).
SSD: `ssd_sgd_lr005`, `ssd_sgd_lr001`, `ssd_adam_lr0005` (по 96 МБ).

Метрики и логи всех прогонов — в `results/logs/`, графики — в `results/plots/`.
