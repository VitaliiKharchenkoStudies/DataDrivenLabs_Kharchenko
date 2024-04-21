import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Введення температурних даних
temps_original = np.array([75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52, 50, 50, 49, 49, 49, 50, 54, 56, 59, 63, 67, 72])
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

# Створення поліноміальних моделей
degree = 10
poly = PolynomialFeatures(degree=degree)

# Метод найменших квадратів
model_ols = make_pipeline(poly, LinearRegression())
model_ols.fit(hours.reshape(-1, 1), temps_original)
temps_ols_pred = model_ols.predict(hours.reshape(-1, 1))
ols_error = np.sqrt(mean_squared_error(temps_original, temps_ols_pred))

# Метод LASSO
model_lasso = make_pipeline(poly, Lasso(alpha=0.1))
model_lasso.fit(hours.reshape(-1, 1), temps_original)
temps_lasso_pred = model_lasso.predict(hours.reshape(-1, 1))
lasso_error = np.sqrt(mean_squared_error(temps_original, temps_lasso_pred))

# Метод Ridge
model_ridge = make_pipeline(poly, Ridge(alpha=0.1))
model_ridge.fit(hours.reshape(-1, 1), temps_original)
temps_ridge_pred = model_ridge.predict(hours.reshape(-1, 1))
ridge_error = np.sqrt(mean_squared_error(temps_original, temps_ridge_pred))

# Метод Elastic Net
model_elastic = make_pipeline(poly, ElasticNet(alpha=0.1, l1_ratio=0.5))
model_elastic.fit(hours.reshape(-1, 1), temps_original)
temps_elastic_pred = model_elastic.predict(hours.reshape(-1, 1))
elastic_error = np.sqrt(mean_squared_error(temps_original, temps_elastic_pred))

# Побудова графіків з різними методами
plt.figure(figsize=(12, 6))

# Поліноміальна апроксимація до "зламу"
plt.subplot(1, 2, 1)
plt.plot(hours, temps_ols_pred, 'r-', label='OLS')
plt.plot(hours, temps_lasso_pred, 'g-', label='LASSO')
plt.plot(hours, temps_ridge_pred, 'b-', label='Ridge')
plt.plot(hours, temps_elastic_pred, 'y-', label='Elastic Net')
plt.scatter(hours, temps_original, color='black', label='Дані')
plt.xlabel("Час")
plt.ylabel("Температура")
plt.title("Поліноміальна апроксимація до \"зламу\"")
plt.legend()

# Злом
temps_broken = temps_original.copy()
np.random.seed(0)
random_index = np.random.choice(hours)
temps_broken[random_index - 1] = 0  # "Злам" температури

# Повторна регресія з оновленими даними
# Метод найменших квадратів
model_ols.fit(hours.reshape(-1, 1), temps_broken)
temps_ols_pred_broken = model_ols.predict(hours.reshape(-1, 1))
ols_error_broken = np.sqrt(mean_squared_error(temps_broken, temps_ols_pred_broken))

# Метод LASSO
model_lasso.fit(hours.reshape(-1, 1), temps_broken)
temps_lasso_pred_broken = model_lasso.predict(hours.reshape(-1, 1))
lasso_error_broken = np.sqrt(mean_squared_error(temps_broken, temps_lasso_pred_broken))

# Метод Ridge
model_ridge.fit(hours.reshape(-1, 1), temps_broken)
temps_ridge_pred_broken = model_ridge.predict(hours.reshape(-1, 1))
ridge_error_broken = np.sqrt(mean_squared_error(temps_broken, temps_ridge_pred_broken))

# Метод Elastic Net
model_elastic.fit(hours.reshape(-1, 1), temps_broken)
temps_elastic_pred_broken = model_elastic.predict(hours.reshape(-1, 1))
elastic_error_broken = np.sqrt(mean_squared_error(temps_broken, temps_elastic_pred_broken))

# Поліноміальна апроксимація після "зламу"
plt.subplot(1, 2, 2)
plt.plot(hours, temps_ols_pred_broken, 'r-', label='OLS')
plt.plot(hours, temps_lasso_pred_broken, 'g-', label='LASSO')
plt.plot(hours, temps_ridge_pred_broken, 'b-', label='Ridge')
plt.plot(hours, temps_elastic_pred_broken, 'y-', label='Elastic Net')
plt.scatter(hours, temps_broken, color='black', label='Дані')
plt.xlabel("Час")
plt.ylabel("Температура")
plt.title("Поліноміальна апроксимація після \"зламу\"")
plt.legend()

plt.tight_layout()
plt.show()

print("Похибка E2(f) до \"зламу\":")
print("OLS:", ols_error)
print("LASSO:", lasso_error)
print("Ridge:", ridge_error)
print("Elastic Net:", elastic_error)

print("Похибка E2(f) після \"зламу\":")
print("OLS:", ols_error_broken)
print("LASSO:", lasso_error_broken)
print("Ridge:", ridge_error_broken)
print("Elastic Net:", elastic_error_broken)