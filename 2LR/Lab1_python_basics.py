import random

# 1. Создаем список, заполненный случайными числами
list_length = 10 # Измените длину списка по желанию
random_list = [random.randint(1, 100) for _ in range(list_length)] # Числа от 1 до 100

# 2. Цикл, суммирующий четные значения
sum_of_evens = 0
for number in random_list:
    if number % 2 == 0:
        sum_of_evens += number

# 3. Вывод суммы на экран
print("Список случайных чисел:", random_list)
print("Сумма четных чисел:", sum_of_evens)

