# Лабораторная работа №1 — Регрессия (EfficientNet + NAdam)

## 1. Теоретическая база
В работе изучается применение нейронных сетей для задачи регрессии на примере датасета автомобилей. Целевая переменная — **MSRP** (розничная цена автомобиля).  
Рассматриваются реализации модели на Numpy, PyTorch и TensorFlow с оптимизаторами **NAdam** и **Adam**.

## 2. Описание системы
Разработано 3 версии EfficientNet-like архитектуры:
- **NumPy** — с ручной реализацией градиентного спуска.
- **PyTorch** — с использованием `torch.optim.NAdam` и `torch.optim.Adam`.
- **TensorFlow** — с `tf.keras.optimizers.Nadam` и `tf.keras.optimizers.Adam`.

Входные данные нормализуются, целевая колонка — `MSRP`.  
Потери измеряются функцией **MSE (Mean Squared Error)**.

## 3. Результаты и тестирование
Пример логов обучения (NumPy):

```
ep 1 loss 5.018
ep 5 loss 0.167
ep 10 loss 0.141
ep 20 loss 0.083
ep 30 loss 0.074
```

В результате:
- **NumPy** — базовый уровень.
- **PyTorch / TensorFlow** — более быстрая сходимость и лучшая точность.

### Графики
После обучения в папке `results/` сохраняются:
- `loss_curve.png` — изменение функции потерь;
- `accuracy_curve.png` — изменение точности (по MSE);
- `report.json` — численные результаты.

## 4. Выводы
1. Модель EfficientNet показывает хорошую сходимость даже при небольшом количестве эпох.
2. Оптимизатор **NAdam** демонстрирует более стабильное уменьшение ошибки по сравнению с классическим Adam.
3. Torch и TensorFlow обеспечивают более быстрое обучение, чем реализация на NumPy.

## 5. Использованные источники
1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*.
2. Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*.
3. TensorFlow и PyTorch официальная документация.
