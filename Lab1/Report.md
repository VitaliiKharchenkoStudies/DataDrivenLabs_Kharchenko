# Лабораторна робота №1
## Data-driven підходи у моделюванні
### Харченко Віталій Андрійович, група ІКМ-М223б
- для перевірки встановіть залежності з файлу requirements.txt
## Аналіз залежності деформації від напруження

Цей код використовується для аналізу залежності деформації від напруження. Він використовує бібліотеки pandas, numpy, matplotlib.pyplot, scipy і sklearn.linear_model.

### Покрокове пояснення коду:

1. **Імпорт бібліотек:**
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate
    from sklearn.linear_model import LinearRegression
    ```

2. **Читання даних з CSV-файлу:**
    ```python
    dataset = pd.read_csv('dataset.csv')
    ```
    Цей код читає дані з CSV-файлу "dataset.csv" та зберігає їх у DataFrame з назвою "dataset".

3. **Отримання значень напруження і деформації:**
    ```python
    stress = dataset.iloc[:, 3] # 4-та колонка (індекс 3)
    strain = dataset.iloc[:, 4] # 5-та колонка (індекс 4)
    ```
    Цей код витягує стовпчики "stress" (напруження) та "strain" (деформація) з DataFrame "dataset".

4. **Видалення дублікатів по осі напруження:**
    ```python
    data = pd.DataFrame({'stress': stress, 'strain': strain}).drop_duplicates(subset='stress')

    stress = data['stress']
    strain = data['strain']
    ```
    Цей код видаляє дублікати рядків DataFrame, де значення "stress" (напруження) однакові. Оновлені значення "stress" (напруження) та "strain" (деформація) зберігаються в нових змінних.

5. **Інтерполяція з кубічним методом:**
    ```python
    f_interp = interpolate.interp1d(stress, strain, kind='cubic', fill_value='extrapolate')
    ```
    Цей код створює об'єкт інтерполяції, який використовує кубічний метод для інтерполяції значень "strain" (деформація) для заданих значень "stress" (напруження).

6. **Лінійна регресія:**
    ```python
    regression = LinearRegression()
    stress_reshaped = stress.values.reshape(-1, 1)
    regression.fit(stress_reshaped, strain)
    strain_pred = regression.predict(stress_reshaped)
    ```
    Цей код використовує метод лінійної регресії для моделювання залежності "strain" (деформація) від "stress" (напруження). 

7. **Побудова графіків:**
    ```python
    plt.figure(figsize=(12, 6))

    # Перший графік: усі дані
    x_interp = np.linspace(min(stress), max(stress), 100)
    y_interp = f_interp(x_interp)

    plt.subplot(1, 3, 1)
    plt.scatter(stress, strain, label='Дані')
    plt.plot(x_interp, y_interp, 'r-', label='Інтерполяція')
    plt.plot(stress, strain_pred, 'b-', label='Лінійна регресія')
    plt.title('Розтягування: усі дані')
    plt.xlabel('Напруження')
    plt.ylabel('Деформація')
    plt.legend()

    # Другий графік: тільки інтерполяція
    plt.subplot(1, 3, 2)
    plt.plot(x_interp, y_interp, 'r-', label='Інтерполяція (кубічна)')
    plt.title('Розтягування: Інтерполяція')
    plt.xlabel('Напруження')
    plt.ylabel('Деформація')

    # Третій графік: лінійна регресія
    plt.subplot(1, 3, 3)
    plt.plot(stress, strain_pred, 'b-', label='Лінійна регресія')
    plt.scatter(stress, strain, color='black', s=10, alpha=0.5, label='Дані')
    plt.title('Розтягування: Лінійна ре


Це виведе графік:
![alt графік](https://media.discordapp.net/attachments/917547349864230912/1231683680456675338/image.png?ex=6637d9d0&is=662564d0&hm=f752ff7478cfcec041c0ae9d07a67c623a6bea1ec43f29b4af6e4d8984878179&=&format=webp&quality=lossless)


А також дані у консоль:


```
Коефіцієнт регресії: [2508.13629493]
Вільний член: 387.1456958366113
```