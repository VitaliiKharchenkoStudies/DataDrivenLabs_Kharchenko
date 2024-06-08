# Лабораторна робота №4
## Виконав: Харченко Віталій, Група ІКМ-М223Б

### Тема: Physics-Informed Neural Networks (PINNs)

У даній лабораторній роботі ми дослідили концепцію Physics-Informed Neural Networks (PINNs) та створили базову модель, яка включає фізичні принципи в процес навчання.

### Опис роботи

#### Кроки виконання роботи:

1. **Імпорт бібліотек:**
   Цей код імпортує необхідні бібліотеки для побудови нейронної мережі (PyTorch) та візуалізації результатів (Matplotlib).

   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   import matplotlib.pyplot as plt
   ```

2. **Побудова моделі:**
   Цей код створює функцію `buildModel`, яка будує нейронну мережу з одним прихованим шаром, використовуючи лінійні шари та активаційну функцію Tanh. Це дозволяє моделі ефективно навчатися нелінійним залежностям.

   ```python
   def buildModel(inDim, hidDim, outDim):
       model = nn.Sequential(
           nn.Linear(inDim, hidDim),  # перший шар
           nn.Tanh(),  # активаційна функція
           nn.Linear(hidDim, outDim)
       )
       return model
   ```

3. **Обчислення градієнтів:**
   Цей код створює функцію `get_derivative`, яка обчислює градієнти (похідні) вихідного значення `y` відносно вхідного значення `x`. Це дозволяє визначати швидкість зміни значень моделі.

   ```python
   def get_derivative(y, x):
       return torch.autograd.grad(y, x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
   ```

4. **Обчислення залишків рівняння:**
   Цей код створює функцію `get_residual`, яка обчислює залишки диференціального рівняння. Це дозволяє моделі враховувати фізичні закони в процесі навчання.

   ```python
   def get_residual(u, x, q, E, A):
       u_x = get_derivative(u, x)
       u_xx = get_derivative(u_x, x)
       return E * A * u_xx + q
   ```

5. **Забезпечення граничних умов:**
   Цей код створює функцію `get_boundary_residual`, яка перевіряє граничні умови на краях стержня. Це дозволяє моделі дотримуватися фізичних обмежень.

   ```python
   def get_boundary_residual(u_pred, u_actual):
       return u_pred - u_actual
   ```

6. **Визначення розмірів моделі:**
   Цей код задає розміри вхідного, прихованого та вихідного шарів нейронної мережі.

   ```python
   inDim = 1
   hidDim = 10
   outDim = 1
   ```

7. **Ініціалізація моделі:**
   Цей код створює модель нейронної мережі, використовуючи функцію `buildModel`. Це дозволяє створити структуру нейронної мережі.

   ```python
   model = buildModel(inDim, hidDim, outDim)
   ```

8. **Визначення оптимізатора:**
   Цей код використовує оптимізатор Adam для навчання моделі. Це дозволяє ефективно мінімізувати функцію втрат.

   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   ```

9. **Задання матеріальних та фізичних властивостей:**
   Цей код визначає фізичні константи, такі як модуль Юнга (E), площа перерізу (A) та розподілене навантаження (q). Це дозволяє моделі враховувати фізичні характеристики стержня.

   ```python
   E = 210e9  # Модуль Юнга в Паскалях
   A = 0.01   # Площа перерізу в квадратних метрах
   q = 1000   # Розподілене навантаження в Н'ютонах на метр
   ```

10. **Задання граничних умов:**
    Цей код встановлює граничні умови для задачі, які визначають зміщення на краях стержня.

    ```python
    u0 = 0  # Зміщення при x=0
    uL = 0.01  # Зміщення при x=L
    ```

11. **Навчання моделі:**
    Цей код реалізує цикл навчання моделі на 1000 епох, обчислюючи втрати (residuals) і оновлюючи ваги моделі. Це дозволяє моделі навчатися і мінімізувати залишки.

    ```python
    num_epochs = 1000

    for epoch in range(num_epochs):
        x = torch.linspace(0, 1, 100).view(-1, 1)
        x.requires_grad = True
        u = model(x)
        res_eq = get_residual(u, x, q, E, A)
        res_bc0 = get_boundary_residual(model(torch.tensor([[0.0]])), torch.tensor([[u0]]))
        res_bcL = get_boundary_residual(model(torch.tensor([[1.0]])), torch.tensor([[uL]]))
        loss = torch.mean(res_eq**2) + res_bc0**2 + res_bcL**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    ```

12. **Візуалізація результатів:**
    Цей код побудує графік, який показує передбачене зміщення уздовж стержня. Це дозволяє візуально оцінити результати моделі.

    ```python
    x_test = torch.linspace(0, 1, 100).view(-1, 1)
    u_test = model(x_test).detach().numpy()
    plt.plot(x_test.numpy(), u_test, label='Predicted displacement')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.show()
    ```

### Результати

Як результат отримуємо графік


![alt графік](https://media.discordapp.net/attachments/917547349864230912/1248938765804634223/image.png?ex=66657c5d&is=66642add&hm=db9273f291111045c72d320061b5516133e4e40e2993c7093884e91edbe8937d&=&format=webp&quality=lossless)