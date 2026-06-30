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

**Датасет:** [Fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia) — 46 781 изображение предметов одежды с разметкой в формате bounding box (лицензия CC-BY-4.0).

**Метрики качества:** mAP, Precision, Recall, F1-score.

## Структура репозитория

```
fashion-object-detection/
├── README.md              # описание проекта
├── requirements.txt       # зависимости
├── setup.py               # установка пакета
├── .gitignore
├── configs/
│   └── default.yaml       # конфигурация экспериментов (гиперпараметры, пути)
├── data/
│   ├── raw/               # исходные данные 
│   └── processed/         # обработанные данные
├── src/
│   ├── dataset/           # загрузка и предобработка данных
│   ├── models/            # реализации моделей
│   ├── training/          # логика обучения
│   ├── evaluation/        # вычисление метрик
│   └── utils/             # вспомогательные функции
├── notebooks/
│   └── exploration.ipynb  # разведочный анализ данных (EDA)
├── results/
│   ├── plots/             # графики
│   └── logs/              # логи обучения
└── main.py                # точка входа
```

## Требования

- Python 3.10+
- PyTorch
- GPU (рекомендуется; проект рассчитан на запуск в Kaggle Notebooks / Google Colab)

## Установка

```bash
git clone https://github.com/<username>/fashion-object-detection.git
cd fashion-object-detection
pip install -r requirements.txt
```

## Запуск

Подготовка данных (загрузка Fashionpedia и предобработка):

```bash
python main.py --prepare-data
```

Обучение выбранной модели:

```bash
python main.py --model yolo
python main.py --model faster_rcnn
python main.py --model ssd
python main.py --model efficientdet
python main.py --model detr
```

Оценка и сравнение результатов сохраняются в директории `results/`.

## Результаты

Итоговое сравнение моделей по метрикам качества:

| Модель | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|--------|---------|--------------|-----------|--------|-----|
| YOLOv8 | — | — | — | — | — |
| Faster R-CNN | — | — | — | — | — |
| SSD | — | — | — | — | — |
| EfficientDet | — | — | — | — | — |
| DETR | — | — | — | — | — |

*Таблица заполняется по результатам экспериментов.*

## Лицензия

Датасет Fashionpedia распространяется по лицензии CC-BY-4.0.

## Автор

Кречетников Артём (БВТ2403) 
Направление «Информатика и вычислительная техника»
МТУСИ
