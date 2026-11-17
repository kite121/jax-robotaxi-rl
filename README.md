# Robotaxi

## Overview

Проект по заданию IAEML: автомобильная среда на JAX и одна нейросеть для карт 0, 1 и 2. Модель проходит короткое обучение по демонстрациям (warmup), затем дообучается методом PPO. Планировщик задаёт маршрут, нейросеть выбирает ускорение и руль.

## Repository structure

- `src/env/` — карты, физика, сенсоры, маршруты и награда.
- `src/algorithms/` — нейросеть, warmup, сбор опыта и PPO.
- `src/utils/` — визуализация и загрузка весов.
- `src/main.py`, `play.py`, `evaluate.py`, `train.py` — ручной режим, просмотр модели, оценка и обучение.
- `models/` — `config.json` и веса до (`pretrained_model.eqx`) и после PPO (`best_model.eqx`).
- `docs/` — [отчёт 1](docs/Report1.pdf) и [отчёт 2](docs/Report2.pdf).
- `videos/` — демонстрация среды и сравнение warmup/PPO.
- `requirements.txt` — закреплённые зависимости.

## Reproduction

Все команды выполняются из корня репозитория. Проверенная версия Python — 3.14.3 на Linux.

**Установка**

```bash
python -m venv .venv
```

Если Python вызывается как `python3` или `py`, используйте это имя при создании окружения. Затем активируйте его:

| Терминал | Команда |
|---|---|
| Linux/macOS, bash или zsh | `source .venv/bin/activate` |
| Windows, PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows, cmd | `.venv\Scripts\activate.bat` |

```bash
python -m pip install -r requirements.txt
```

**Просмотр**

```bash
python src/main.py --map-id 0
python src/play.py models --map-id 0
```

Первая команда — ручное управление, вторая — готовая модель. Карты: `0`, `1`, `2`; для модели до PPO добавьте к `play.py` параметр `--weights pretrained`.
В ручном режиме W/S — газ и торможение/задний ход, A/D — руль. В обоих режимах: пробел — пауза, R — сброс, L — лучи, Esc — выход.

**Оценка всех карт и сравнение до/после PPO**

```bash
python src/evaluate.py models --episodes 1
python src/evaluate.py models --weights pretrained --episodes 50 --stochastic --seed 321
python src/evaluate.py models --weights best --episodes 50 --stochastic --seed 321
```

Первая проверка выполняется без шума; следующие — по 50 эпизодов на карту с шумом действий и одинаковым seed оценки. Результаты сохраняются в CSV/JSON. Подробные результаты и ограничения описаны в отчёте 2.

**Повторное обучение с нуля на трёх картах**

```bash
python src/train.py --config models/config.json --output-dir results/my_run
```

Конфигурация задаёт 1500 обновлений warmup и 786432 решения PPO. Каталог нового запуска `results/my_run/maps0_1_2_seed0` должен отсутствовать; туда сохранятся веса и журналы. Для просмотра и оценки новой модели укажите этот путь вместо `models`. Первая итерация включает компиляцию JAX.
