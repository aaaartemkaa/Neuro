import torch

# 1. Создаем тензор x целочисленного типа со случайным значением
x = torch.randint(1, 10, (1,)) 

# 2. Преобразуем тензор к типу float32
x = x.to(torch.float32)
x.requires_grad = True

# 3. Проводим ряд операций 
n_even = 2
x_even = x**n_even
x_even = x_even * torch.randint(1, 11, (1,)).to(torch.float32) # случайное число от 1 до 10
x_even = torch.exp(x_even)

# 4. Вычисляем и выводим производную 
x_even.backward()
print(f"Производная для n={n_even}: {x.grad}")