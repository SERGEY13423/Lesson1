"""
homework_autograd.py
Выполнение Задания 2: Автоматическое дифференцирование (Autograd) в PyTorch.
"""

import math
import torch


def task_2_1_simple_gradients():
    """2.1 Простые вычисления с градиентами и аналитическая проверка."""
    print("--- 2.1 Простые вычисления с градиентами ---")

    # Создаем тензоры-скаляры с отслеживанием градиентов
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)

    # Задаем функцию: f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y*z
    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z

    # Запускаем обратное распространение ошибки
    f.backward()

    # Полученные градиенты в PyTorch
    grad_x_torch = x.grad.item()
    grad_y_torch = y.grad.item()
    grad_z_torch = z.grad.item()

    # Аналитический расчет:
    # df/dx = 2*x + 2*y*z
    # df/dy = 2*y + 2*x*z
    # df/dz = 2*z + 2*x*y
    x_val, y_val, z_val = x.item(), y.item(), z.item()
    grad_x_analyt = 2 * x_val + 2 * y_val * z_val
    grad_y_analyt = 2 * y_val + 2 * x_val * z_val
    grad_z_analyt = 2 * z_val + 2 * x_val * y_val

    print(f"f(x,y,z) = {f.item()}")
    print(
        f"df/dx: PyTorch = {grad_x_torch}, Аналитически = {grad_x_analyt} (Совпало: {math.isclose(grad_x_torch, grad_x_analyt)})")
    print(
        f"df/dy: PyTorch = {grad_y_torch}, Аналитически = {grad_y_analyt} (Совпало: {math.isclose(grad_y_torch, grad_y_analyt)})")
    print(
        f"df/dz: PyTorch = {grad_z_torch}, Аналитически = {grad_z_analyt} (Совпало: {math.isclose(grad_z_torch, grad_z_analyt)})\n")


def task_2_2_mse_loss():
    """2.2 Градиент функции потерь MSE по параметрам w и b."""
    print("--- 2.2 Градиент функции потерь MSE ---")

    # Исходные тренировочные данные: x и истинный y_true
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])  # истинная зависимость y = 2x + 1

    # Параметры модели, которые нужно оптимизировать (requires_grad=True)
    w = torch.tensor(1.0, requires_grad=True)  # начальный вес
    b = torch.tensor(0.0, requires_grad=True)  # начальный сдвиг

    # Линейная модель: y_pred = w * x + b
    y_pred = w * x + b

    # Функция потерь MSE: (1/n) * sum((y_pred - y_true)^2)
    # В PyTorch можно посчитать формулой или torch.mean((y_pred - y_true)**2)
    mse_loss = torch.mean((y_pred - y_true) ** 2)

    # Вычисляем градиенты
    mse_loss.backward()

    print(f"Значение MSE: {mse_loss.item():.4f}")
    print(f"Градиент d(MSE)/dw: {w.grad.item():.4f}")
    print(f"Градиент d(MSE)/db: {b.grad.item():.4f}")
    print("Градиенты показывают, в какую сторону нужно корректировать w и b.\n")


def task_2_3_chain_rule():
    """2.3 Цепное правило: дифференцирование f(x) = sin(x^2 + 1)."""
    print("--- 2.3 Цепное правило ---")

    x = torch.tensor(1.5, requires_grad=True)

    # f(x) = sin(x^2 + 1)
    f = torch.sin(x ** 2 + 1)

    # Способ 1: через torch.autograd.grad
    # возвращает кортеж градиентов для переданных входов
    grad_autograd = torch.autograd.grad(outputs=f, inputs=x, retain_graph=True)[0]

    # Способ 2: через стандартный .backward()
    f.backward()
    grad_backward = x.grad

    # Аналитический расчет: f'(x) = cos(x^2 + 1) * 2x
    x_val = x.item()
    grad_analyt = math.cos(x_val ** 2 + 1) * (2 * x_val)

    print(f"Градиент через autograd.grad: {grad_autograd.item():.6f}")
    print(f"Градиент через f.backward():    {grad_backward.item():.6f}")
    print(f"Градиент аналитический:         {grad_analyt:.6f}")
    print(f"Сходимость: {math.isclose(grad_autograd.item(), grad_analyt, rel_tol=1e-5)}\n")


if __name__ == "__main__":
    task_2_1_simple_gradients()
    task_2_2_mse_loss()
    task_2_3_chain_rule()