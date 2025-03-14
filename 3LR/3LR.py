import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 1. Загрузка данных и обработка меток
data = pd.read_csv("data.csv", header=None)
X = data.iloc[:, :-1].values.astype(np.float32)
y = data.iloc[:, -1].values
y = np.where(y == 'Iris-setosa', 0, 1)  
y = y.astype(np.longlong)

# 2. Разделение на обучающую и тестовую выборки
train_size = int(0.7 * len(X))#70%-обучение, 30%-тестирование
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Преобразование в тензоры PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 4. Линейная модель
model = nn.Linear(4, 2)

# 5. Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 6. Обучение модели
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Эпоха {epoch + 1}/{epochs}, Потеря: {loss.item():.4f}')

# 7. Тестирование модели
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1) # torch.max возвращает значения и индексы
    correct = (predicted == y_test).sum().item()
    total = len(y_test)
    accuracy = 100 * correct / total
    print(f'Точность: {accuracy:.2f}%')