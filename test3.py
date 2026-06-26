import torch
import torch.nn as nn
import torch.optim as optim

# 2. Определение архитектуры нейросети
class SimpleNet(nn.Module): # nn.Module - базовый класс для всех нейронных сетей в PyTorch
    def __init__(self, input_size=5, hidden_size=10, output_size=1):
        # super(SimpleNet, self).__init__()
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size) # Скрытый слой: из 5 признаков делаем 10 нейронов. nn.Linear(n,m) создает слой с n входами и m выходами
        self.relu = nn.ReLU() # Функция активации
        self.output = nn.Linear(hidden_size, output_size) # Выходной слой: из 10 нейронов получаем 1 число (регрессия)
        
    def forward(self, x):
        # Описываем, как данные текут по графу вычислений вперед
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

model = SimpleNet() # Экземпляр модели

X_train = torch.randn(100, 5) # 100 объектов и 5 признаков
y_train = (X_train.sum(dim=1, keepdim=True) ** 2) + 3.0 

criterion = nn.MSELoss() # Функция потерь для регрессии

optimizer = optim.Adam(model.parameters(), lr=0.01) # Оптимизатор optim.Adam(веса, скорость обучения)

epochs = 200  # Сколько раз модель просмотрит данные

for epoch in range(epochs):
    model.train() # Перевод модели в режим обучения
    
    optimizer.zero_grad() # Очищаем старые градиенты, накопленные на предыдущем шаге
    
    predictions = model(X_train) # Forward pass. Передаем данные в модель, она строит граф вычислений и выдает предсказания
    
    loss = criterion(predictions, y_train) # Сравниваем предсказания с правильными ответами
    
    loss.backward() # Backpropagation. # Раскручиваем граф вычислений назад, считая производные(градиенты) для каждого веса (.grad)
    
    optimizer.step() # Обновляем веса модели. Оптимизатор Adam берет сохраненные .grad и корректирует веса модели
    
    # Печатаем прогресс каждые 20 эпох
    if (epoch + 1) % 20 == 0:
        print(f"Эпоха [{epoch+1}/{epochs}] | Ошибка (Loss): {loss.item():.4f}")