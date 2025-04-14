# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
def main():
    # Устройство для вычислений
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Преобразования для изображений
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Загрузка собственного набора данных
    train_dataset = torchvision.datasets.ImageFolder(root='./my_data/train',
                                                   transform=data_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root='./my_data/test',
                                                 transform=data_transforms)
    # Названия классов
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Классы в наборе данных: {class_names}")
    print(f"Количество классов: {num_classes}")
    # DataLoader
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                       shuffle=False, num_workers=0)
    # Загрузка предобученной модели ResNet18
    net = torchvision.models.resnet18(pretrained=True)
    # Замораживаем веса
    for param in net.parameters():
        param.requires_grad = False
    # Заменяем последний слой для нашего количества классов
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, num_classes)
    net = net.to(device)
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)
    # Обучение
    num_epochs = 30
    save_loss = []
    print("Начало обучения...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            save_loss.append(loss.item())
            if i % 10 == 9:
                print(f'Эпоха {epoch + 1}, шаг {i + 1}, loss: {running_loss / 10:.3f}')
                running_loss = 0.0
    # Визуализация loss
    plt.figure()
    plt.plot(save_loss)
    plt.title('График функции потерь')
    plt.xlabel('Шаг')
    plt.ylabel('Loss')
    plt.show()
    # Оценка точности
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Точность на тестовом наборе: {100 * correct / total:.2f}%')
    # Сохранение модели
    torch.save(net.state_dict(), 'my_model.pt')
    print("Модель сохранена как 'my_model.pt'")
if __name__ == '__main__':
    freeze_support()
    main()