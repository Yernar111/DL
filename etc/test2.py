import torch
import torch.nn as nn
import torch.optim as optim

X_train = torch.randn(100, 5) # 100 объектов и 5 признаков
# Настоящий таргет (просто для примера добавим нелинейности)
y_train = (X_train.sum(dim=1, keepdim=True) ** 2) + 3.0

# Параметры скрытого слоя

# Веса для скрытого слоя(в виде тензора 5х10). В нейронных сетях веса изначально случайны. Каждый столбец матрицы весов для соответстветствующего нейрона.
w_hidden = torch.randn(5, 10, requires_grad=True) # requires_grad=True говорит PyTorch, что нужно хранить градиенты для этого тензора. В этом слое у нас 5 входов и 10 нейронов, поэтому весов будет 5*10. 

# bias для скрытого слоя(в виде тензора из нулей 1х10). Такой размер обусловлен тем, что для каждого нейрона скрытого слоя соответствует один bias
b_hidden = torch.zeros(1, 10, requires_grad=True) # Сдвиги(bias) изначально зануляем. 

# Параметры выходного слоя

# Веса выходного слоя(в виде тензора случайных чисел 10х1).
w_output = torch.randn(10, 1, requires_grad=True) # 10 нейронов скрытого слоя с одним выходом

# Один единственный bias
b_output = torch.zeros(1, 1, requires_grad=True)

lr=0.01 # Задаем скорость обучения (learning rate)

epochs = 200

for epoch in range(epochs):
    
    # --- ПРЯМОЙ ХОД (FORWARD PASS) ВРУЧНУЮ ---

    z_hidden = X_train @ w_hidden + b_hidden # Форма результата: [100, 5] @ [5, 10] + [1, 10] -> [100, 10]
    
    a_hidden = torch.clamp(z_hidden, min=0) # Функция активации ReLU вручную: зануляем всё, что меньше 0
    
    predictions = a_hidden @ w_output + b_output # Форма результата: [100, 10] @ [10, 1] -> [100, 1]
    
    # --- РАСЧЕТ ОШИБКИ (MSE LOSS) ВРУЧНУЮ ---

    loss = torch.mean((predictions - y_train) ** 2) # Функция потерь
    
    loss.backward() # Backpropagation. Раскручиваем граф вычислений назад, считая производные(градиенты) для каждого веса (.grad)
    
    # --- ОБНОВЛЕНИЕ ВЕСОВ --- (то что происходит при optimizer.step())
    with torch.no_grad(): # Отключаем градиенты на момент изменения весов, чтобы не зациклить граф
        # Классический шаг градиентного спуска (SGD): w = w - lr * grad
        w_hidden -= lr * w_hidden.grad # .grad хранит градиенты тензора, посчитанные в loss.backward()
        b_hidden -= lr * b_hidden.grad
        w_output -= lr * w_output.grad
        b_output -= lr * b_output.grad
        
        # --- РУЧНОЕ ОБНУЛЕНИЕ ГРАДИЕНТОВ ---
        w_hidden.grad.zero_()
        b_hidden.grad.zero_()
        w_output.grad.zero_()
        b_output.grad.zero_()
        
    # Печатаем прогресс каждые 20 эпох
    if (epoch + 1) % 20 == 0:
        print(f"Эпоха [{epoch+1}/{epochs}] | Ошибка (Loss): {loss.item():.4f}")