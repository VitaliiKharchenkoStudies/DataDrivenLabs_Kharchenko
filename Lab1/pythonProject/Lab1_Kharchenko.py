import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.linear_model import LinearRegression

# Читання даних з CSV
dataset = pd.read_csv('dataset.csv')

# Отримання значень напруження і деформації
stress = dataset.iloc[:, 3]  # 4-та колонка (індекс 3)
strain = dataset.iloc[:, 4]  # 5-та колонка (індекс 4)

# Видалення дублікатів по осі напруження
data = pd.DataFrame({'stress': stress, 'strain': strain}).drop_duplicates(subset='stress')

# Оновлення значень після видалення дублікатів
stress = data['stress']
strain = data['strain']

# Інтерполяція з кубічним методом
f_interp = interpolate.interp1d(stress, strain, kind='cubic', fill_value='extrapolate')

# Лінійна регресія
regression = LinearRegression()
stress_reshaped = stress.values.reshape(-1, 1)
regression.fit(stress_reshaped, strain)
strain_pred = regression.predict(stress_reshaped)

# Побудова графіків
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
plt.title('Розтягування: Лінійна регресія')
plt.xlabel('Напруження')
plt.ylabel('Деформація')
plt.legend()

# Відображення графіків
plt.tight_layout()
plt.show()

# Вивід результатів у консоль
print("Коефіцієнт регресії:", regression.coef_)
print("Вільний член:", regression.intercept_)
