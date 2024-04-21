# Лабораторна робота №2
## Data-driven підходи у моделюванні
### Харченко Віталій Андрійович, група ІКМ-М223б


- для перевірки встановіть залежності з файлу requirements.txt

### Хід роботи


```python
np.random.seed(0)
x0 = np.random.uniform(0, 10, 100)
x1 = np.random.uniform(0, 10, 100)
noise = np.random.normal(0, 0.1, 100)  # Шум у даних

y = np.sin(x0) - np.log(x0 + x1) + noise
```


Цей код створює два набори даних x0 та x1, а також додає до них шум. Далі формується цільова змінна y за формулою: y = sin(x0) - log(x0 + x1) + noise.


```python
data = pd.DataFrame({
    'x0': x0,
    'x1': x1,
    'y': y
})
```

Цей код об'єднує дані x0, x1 та y у DataFrame для зручного використання.


```python
X_train, X_test, y_train, y_test = train_test_split(data[['x0', 'x1']], data['y'], test_size=0.3, random_state=42)
```


Цей код ділить дані на навчальний та тестовий набори в пропорції 70/30. Навчальний набір буде використовуватися для тренування моделі, а тестовий - для оцінки її точності.


```python
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
```


Цей код створює об'єкт класу SymbolicRegressor з gplearn. Він налаштовує різні параметри, такі як розмір популяції, кількість поколінь, функції, що використовуються, та критерій зупинки.


```python
model.fit(X_train, y_train)
```


Цей код тренує модель на даних навчального набору.


```python
found_expression = model._program
```


Цей код отримує вираз, який модель gplearn знайшла для опису залежності y від x0 та x1.


```python
print("Вираз, знайдений символьною регресією:")
print(found_expression)
```


Цей код виводить знайдений вираз у читабельній формі, щоб користувач міг його зрозуміти.


```python
y_pred = model.predict(X_test)  # Передбачення
```


```python
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
```


Виведення графіку та дерева


Графік


![alt графік](https://media.discordapp.net/attachments/917547349864230912/1231712880081244230/image.png?ex=6626d182&is=66258002&hm=912dc9e48a73ec49138c8de46d8f4e362d4d0b316b87e25f83695ccf1e490cca&=&format=webp&quality=lossless)


Дерево


![alt дерево](https://media.discordapp.net/attachments/917547349864230912/1231699863360766072/image.png?ex=6637e8e3&is=662573e3&hm=75cf730183497aa0fe39d3748e9ce8d81ee2ca872b4bea397e1fd977a23098be&=&format=webp&quality=lossless&width=567&height=671)