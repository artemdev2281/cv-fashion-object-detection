# Fashion Object Detection

Сравнительный анализ современных моделей детектирования объектов (object detection) применительно к задаче обнаружения предметов одежды и аксессуаров на изображениях.
Проект выполнен в рамках учебной практики (МТУСИ).

## Описание

Цель проекта - сравнить пять современных моделей детектирования объектов на задаче поиска предметов одежды и определить наиболее эффективную для применения в системах визуального поиска (visual search).

**Сравниваемые модели:**
- YOLOv8 (one-stage)
- Faster R-CNN (two-stage)
- SSD (one-stage)
- EfficientDet (efficiency-oriented)
- DETR (transformer-based)

**Датасет:** [Fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia) - 46 781 изображение предметов одежды с разметкой в формате bounding box (лицензия CC-BY-4.0). Из 46 категорий отобрано 11 самостоятельных предметов одежды и аксессуаров.

**Метрики качества:** mAP, Precision, Recall, F1-score.

## Структура репозитория

```
cv-fashion-object-detection/
├── README.md              # описание проекта
├── requirements.txt       # зависимости
├── setup.py               # установка пакета
├── .gitignore
├── configs/
│   └── default.yaml       # конфигурация экспериментов (гиперпараметры, пути)
├── data/                  # данные (содержимое не хранится в git)
│   ├── raw/               # исходные данные
│   └── processed/         # обработанные данные (YOLO + COCO форматы)
├── src/
│   ├── dataset/           # загрузка и предобработка данных (Fashionpedia, COCO)
│   ├── models/            # реализации моделей (yolo, faster_rcnn, ssd, efficientdet, detr)
│   ├── training/          # логика обучения
│   ├── evaluation/        # вычисление метрик (COCOeval)
│   └── utils/             # утилиты: конфигурация, логирование, визуализация
├── notebooks/             # ноутбуки Kaggle
│   ├── exploration.ipynb              # разведочный анализ данных (EDA)
│   ├── train_yolov8.ipynb             # обучение YOLOv8
│   ├── train_torchvision.ipynb        # обучение Faster R-CNN и SSD
│   ├── train_efficientdet_detr.ipynb  # обучение EfficientDet и DETR
│   └── hyperparameter_experiments.ipynb  # подбор гиперпараметров (YOLO, SSD)
├── results/
│   ├── plots/             # графики (loss-кривые, сравнения, per-class)
        └── detections/    # сравнение работы моделей на примерах изображений
│   ├── logs/              # логи обучения и метрики (results.csv, metrics.json)
│   └── weights/           # веса моделей (локально, не в git - см. ниже)
└── main.py                # точка входа
```

## Требования

- Python 3.10+
- PyTorch, torchvision
- Для EfficientDet и DETR — дополнительно `effdet`, `timm`, `transformers` (в `requirements.txt`).
- GPU (рекомендуется; проект рассчитан на запуск в Kaggle Notebooks / Google Colab)

## Установка

```bash
git clone https://github.com/neuromindgpt/cv-fashion-object-detection.git
cd cv-fashion-object-detection
pip install -r requirements.txt
```

## Запуск

Подготовка данных (загрузка Fashionpedia и предобработка - фильтрация 11 классов, разбиение train/val/test, экспорт в форматы YOLO и COCO):

```bash
python main.py --prepare-data
```

Обучение выбранной модели (параметры берутся из `configs/default.yaml`):

```bash
python main.py --model yolo
python main.py --model faster_rcnn
python main.py --model ssd
python main.py --model efficientdet
python main.py --model detr
```

Объём обучающей подвыборки задаётся параметром `training.subset_size` в `configs/default.yaml` (по умолчанию 3000 - как в экспериментах; `null` = полный train). Итоговая оценка всегда выполняется на полном тестовом множестве.

Построение графиков из логов обучения:

```bash
python -m src.utils.utils        # графики сохраняются в results/plots/
```

Обучение выполнялось в среде Kaggle Notebooks (GPU Tesla T4); соответствующие ноутбуки — в каталоге `notebooks/`. Результаты и метрики сохраняются в директории `results/`.

## Результаты

Итоговое сравнение моделей по метрикам качества на тестовом множестве (4437 изображений; обучение — на подвыборке 3000 изображений, 20 эпох, seed 42):

| Модель | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Время обучения (T4) |
|--------|---------|--------------|-----------|--------|-----|---------------------|
| Faster R-CNN | **0,565** | 0,384 | 0,636 | **0,730** | **0,680** | ~2,3 ч |
| YOLOv8n | 0,534 | **0,390** | 0,581 | 0,537 | 0,558 | ~12 мин |
| SSD | 0,413 | 0,243 | 0,640 | 0,549 | 0,591 | ~22 мин |
| EfficientDet | 0,370 | 0,177 | **0,826** | 0,347 | 0,489 | ~44 мин |
| DETR | 0,214 | 0,112 | 0,340 | 0,437 | 0,382 | ~5,7 ч |

Наивысшую точность по mAP@0.5 показала Faster R-CNN; наилучший компромисс точность/скорость обеспечивает YOLOv8n (сопоставимая точность при обучении примерно в 11 раз быстрее). Графики (кривые обучения, сравнения, детализация по классам) — в `results/plots/`.

## Воспроизводимость

Воспроизводимость обеспечена на уровне протокола: фиксируется random seed (`training.seed = 42`), для всех моделей используются одна и та же обучающая подвыборка и одно и то же тестовое множество, метрики считаются единым модулем (`src/evaluation/metrics.py`). Полная побитовая идентичность результатов между запусками не гарантируется из-за недетерминированности GPU-вычислений (cuDNN) и зависимости от аппаратной платформы; расхождение метрик обычно в пределах ±1 процентного пункта.

## Веса моделей

Файлы весов (`*.pt`, `*.pth`) в git не хранятся (см. `.gitignore`) из-за большого размера (Faster R-CNN и DETR — по ~159 МБ, что превышает лимит GitHub 100 МБ/файл). Они сохраняются локально в `results/weights/` и в выводе соответствующих Kaggle-ноутбуков. Метрики, логи обучения и графики (`results/logs/`, `results/plots/`) хранятся в репозитории.

## Лицензия

Датасет Fashionpedia распространяется по лицензии CC-BY-4.0.

## Автор

Кречетников Артём (БВТ2403)
Направление «Информатика и вычислительная техника»
Профиль «Искусственный интеллект и машинное обучение» 
МТУСИ
