import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# 3 вариант
# 1.Загрузка и подготовка данных
df = pd.read_csv('dataset_simple.csv')
X = torch.tensor(df[['age', 'income']].values, dtype=torch.float32)
y = torch.tensor(df['will_buy'].values, dtype=torch.float32).reshape(-1, 1)
# 2.Определение нейронной сети
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 3)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.tanh(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x
# 3.Создание экземпляра сети и настройка обучения
net = SimpleNet()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# 4.Обучение сети с сохранением истории потерь
epochs = 100
losses = []
for epoch in range(epochs):
    pred = net(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Эпоха {epoch+1}, Потери: {loss.item():.4f}')
# 5.Оценка и вывод результатов
with torch.no_grad():
    pred = net(X)
    pred_labels = (pred >= 0.5).float()
    accuracy = (pred_labels == y).float().mean()
    print(f'\nТочность: {accuracy.item():.4f}')
# 6.Визуализация результатов
plt.figure(figsize=(10, 5))
#7.График потерь
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), losses)
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.title("График потерь во время обучения")