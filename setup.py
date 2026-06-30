"""Установка пакета проекта Fashion Object Detection."""

from pathlib import Path

from setuptools import find_packages, setup

_here = Path(__file__).parent
_requirements = (_here / "requirements.txt").read_text(encoding="utf-8").splitlines()
_install_requires = [
    line.strip()
    for line in _requirements
    if line.strip() and not line.startswith("#")
]

setup(
    name="cv-fashion-object-detection",
    version="0.1.0",
    description=(
        "Сравнительный анализ моделей детектирования объектов одежды "
        "на датасете Fashionpedia"
    ),
    author="Кречетников",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=_install_requires,
)
