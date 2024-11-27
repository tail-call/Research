import numpy as np
import matplotlib.pyplot as plt

# Настройки для генерации случайных данных
np.random.seed(42)
n = 100  # количество элементов
gradients = np.random.randn(n) * 10  # случайные градиенты с большой вариацией (среднее 0, std = 10)

# Порог для обрезки градиентов
clip_value = 3

# Обрезка градиентов
clipped_gradients = np.clip(gradients, -clip_value, clip_value)

# Визуализация до и после обрезки
plt.figure(figsize=(10, 6))

# График для оригинальных градиентов
plt.subplot(1, 2, 1)
plt.plot(gradients, label='Original Gradients', color='blue')
plt.axhline(y=clip_value, color='r', linestyle='--', label='Clip Threshold')
plt.axhline(y=-clip_value, color='r', linestyle='--')
plt.title('Original Gradients')
plt.xlabel('Index')
plt.ylabel('Gradient Value')
plt.legend()

# График для обрезанных градиентов
plt.subplot(1, 2, 2)
plt.plot(clipped_gradients, label='Clipped Gradients', color='green')
plt.axhline(y=clip_value, color='r', linestyle='--', label='Clip Threshold')
plt.axhline(y=-clip_value, color='r', linestyle='--')
plt.title('Clipped Gradients')
plt.xlabel('Index')
plt.ylabel('Gradient Value')
plt.legend()

plt.tight_layout()
plt.show()
