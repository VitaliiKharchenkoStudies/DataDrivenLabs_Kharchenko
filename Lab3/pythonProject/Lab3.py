import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
import graphviz  # Бібліотека для візуалізації дерев
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Створення випадкових даних для регресії
np.random.seed(0)
x0 = np.random.uniform(0, 10, 100)
x1 = np.random.uniform(0, 10, 100)
noise = np.random.normal(0, 0.1, 100)  # Шум у даних

# Генерація даних за формулою
y = np.sin(x0) - np.log(x0 + x1) + noise

# Об'єднання даних у DataFrame
data = pd.DataFrame({
    'x0': x0,
    'x1': x1,
    'y': y
})

# Поділ даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(data[['x0', 'x1']], data['y'], test_size=0.3, random_state=42)

# Створення символьної регресії з gplearn
model = SymbolicRegressor(
    population_size=5000,
    generations=20,
    tournament_size=20,
    stopping_criteria=0.01,
    const_range=(-1, 1),
    init_depth=(2, 6),
    function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log'),
    metric='mse',
    parsimony_coefficient=0.1,
    random_state=0
)

# Тренування моделі
model.fit(X_train, y_train)

# Отримання виразу, знайденого символьною регресією
found_expression = model._program

# Вивід знайденого виразу у читабельній формі
print("Вираз, знайдений символьною регресією:")
print(found_expression)

# Аналіз знайденого виразу та порівняння його з очікуваним
y_pred = model.predict(X_test)  # Передбачення на тестових даних
mse = mean_squared_error(y_test, y_pred)

print("Середньоквадратична похибка на тестових даних:", mse)

# Побудова графіка для оригінальних і передбачених даних
plt.scatter(data['x0'], data['y'], color='blue', label='Оригінальні дані')
plt.scatter(X_test['x0'], y_pred, color='red', label='Передбачені дані')
plt.xlabel("x0")
plt.ylabel("y")
plt.legend()
plt.title("Оригінальні та передбачені дані")
plt.show()

# Отримання дерева виразу у форматі Graphviz
expr_tree = model._program.export_graphviz()

# Візуалізація дерева за допомогою graphviz
graph = graphviz.Source(expr_tree)
graph.render("symbolic_regression_tree", format='png', cleanup=False)  # Зберігає у PNG
graph.view()
