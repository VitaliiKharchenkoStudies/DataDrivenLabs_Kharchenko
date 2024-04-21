import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Дані за 24-годинний цикл
temps = np.array([75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52, 50, 50, 49, 49, 49, 50, 54, 56, 59, 63, 67, 72])
times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

# Інтерполяція
x_interp = np.arange(1, 24.01, 0.01)  # Крок 0.01 для більшої точності
f_linear_interp = interpolate.interp1d(times, temps, kind='linear', fill_value='extrapolate')
f_spline_interp = interpolate.interp1d(times, temps, kind='cubic', fill_value='extrapolate')

# Косинусне апроксимування: y = A * cos(B * x) + C
def cos_model(x, A, B, C):
    return A * np.cos(B * x) + C

# Оптимальні значення для A, B, C
popt, _ = optimize.curve_fit(cos_model, times, temps, p0=[10, 0.1, 50])
A, B, C = popt
temps_cos_pred = cos_model(times, A, B, C)  # Прогноз на основі косинусного апроксимування

# Лінійна регресія
times_squared = times ** 2
X = np.column_stack((times_squared, times, np.ones_like(times)))
reg = LinearRegression()
reg.fit(X, temps)
temps_parabola_pred = reg.predict(X)

# Обчислення похибки E2(f) для всіх методів
E2_f_cos = np.sqrt(np.mean((temps_cos_pred - temps) ** 2))
E2_f_linear_interp = np.sqrt(np.mean((f_linear_interp(x_interp) - f_spline_interp(x_interp)) ** 2))
E2_f_spline_interp = np.sqrt(np.mean((f_spline_interp(x_interp) - f_linear_interp(x_interp)) ** 2))
E2_f_parabola = np.sqrt(np.mean((temps_parabola_pred - temps) ** 2))

# Побудова графіків
plt.figure(figsize=(12, 10))

# Перший ряд графіків
plt.subplot(2, 3, 1)
plt.scatter(times, temps, color='black', label='Температурні дані')
plt.plot(times, temps_cos_pred, 'r-', label=f'Косинусне апроксимування: y = {A:.2f} * cos({B:.2f} * x) + {C:.2f}')
plt.xlabel('Час (години)')
plt.ylabel('Температура (°F)')
plt.title('Температура: Косинусне апроксимування')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(x_interp, f_linear_interp(x_interp), 'b-', label='Лінійна інтерполяція')
plt.plot(x_interp, f_spline_interp(x_interp), 'g-', label='Сплайн інтерполяція')
plt.xlabel('Час (години)')
plt.ylabel('Температура (°F)')
plt.title('Температура: Інтерполяція')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(times, temps_parabola_pred, 'r-', label='Параболічне апроксимування')
plt.scatter(times, temps, color='black', s=10, alpha=0.5, label='Дані')
plt.xlabel('Час (години)')
plt.ylabel('Температура (°F)')
plt.title('Температура: Параболічне апроксимування')
plt.legend()

# Другий ряд графіків
plt.subplot(2, 3, 4)
plt.plot(times, temps, 'k-', label='Оригінальні дані')
plt.xlabel('Час (години)')
plt.ylabel('Температура (°F)')
plt.title('Температура: Оригінальні дані')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(x_interp, f_spline_interp(x_interp), 'g-', label='Сплайн інтерполяція')
plt.xlabel('Час (години)')
plt.ylabel('Температура (°F)')
plt.title('Температура: Сплайн інтерполяція')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(x_interp, f_linear_interp(x_interp), 'b-', label='Лінійна інтерполяція')
plt.xlabel('Час (години)')
plt.ylabel('Температура (°F)')
plt.title('Температура: Лінійна інтерполяція')
plt.legend()

plt.tight_layout()
plt.show()

# Вивід похибок
print("Похибка E2(f) для косинусного апроксимування:", E2_f_cos)
print("Похибка E2(f) для параболічного апроксимування:", E2_f_parabola)
print("Похибка E2(f) для лінійної інтерполяції:", E2_f_linear_interp)
print("Похибка E2(f) для сплайн інтерполяції:", E2_f_spline_interp)
