import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv')
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1) # Классификация на 2 класса (setosa vs others)
X = df.iloc[:, [0, 2, 3]].values


def neuron(w, x):
    return 1 if np.dot(w, x) >= 0 else -1


w = np.random.rand(4) # Случайная инициализация весов (свободный член + 3 признака)
eta = 0.1 # Скорость обучения
epochs = 10 # Количество эпох обучения

w_history = [] #Список для хранения весов на каждой итерации

for epoch in range(epochs):
    errors = 0
    for xi, target in zip(X, y):
        x_extended = np.concatenate(([1], xi)) # Добавляем свободный член
        prediction = neuron(w, x_extended)
        update = eta * (target - prediction)
        w += update * x_extended
        errors += int(update != 0) #Подсчет количества неверных классификаций.
    w_history.append(w.copy())
    print(f"Epoch {epoch+1}/{epochs}, Errors: {errors}")

X_2d = X[:,:2]

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(X_2d[y == -1, 0], X_2d[y == -1, 1], color='blue', marker='x', label='Other')

xl = np.linspace(min(X_2d[:, 0]), max(X_2d[:, 0]), 100)

for i, w in enumerate(w_history):
    if w[2] != 0: #Проверка на случай деления на ноль.
        yl = -(xl * w[1] + w[0]) / w[2]
        plt.plot(xl, yl, label=f'Epoch {i+1}')

plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Perceptron Learning Rule for Iris Classification')
plt.legend()
plt.show()
