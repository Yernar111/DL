import torch.nn as nn

# import torch.nn.functional as F
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         # Вход: [Батч, 1, 28, 28] (1 канал — ч/б картинка)
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) # Выход первого слоя: [Батч, 16, 28, 28]
        
#         # Вход: [Батч, 16, 14, 14] (после MaxPool2d размер уменьшится в 2 раза)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # Выход второго слоя: [Батч, 32, 14, 14]
        
#         # Слой пулинга: уменьшает высоту и ширину в 2 раза (окно 2х2)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Полносвязные слои (Dense / Linear)
#         # После двух пулингов размер картинки 28x28 станет 7x7. Каналов стало 32.
#         # Итого на входе: 32 * 7 * 7 = 1568 элементов
#         self.fc1 = nn.Linear(32 * 7 * 7, 128)
#         self.dropout = nn.Dropout(p=0.25)
#         self.fc2 = nn.Linear(128, 10) # 10 классов (цифры от 0 до 9)

#     def forward(self, x):
#         # Первый блок: Свертка -> ReLU -> Пулинг (28x28 -> 14x14)
#         x = self.pool(F.relu(self.conv1(x)))
#         # Второй блок: Свертка -> ReLU -> Пулинг (14x14 -> 7x7)
#         x = self.pool(F.relu(self.conv2(x)))
        
#         # Выпрямляем тензор (Flatten) перед подачей в полносвязный слой
#         # Превращаем форму [Батч, 32, 7, 7] в [Батч, 1568]
#         x = x.view(-1, 32 * 7 * 7)
        
#         # Полносвязный слой с дропаутом
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        
#         # Выходной слой (логиты, Softmax зашит внутри CrossEntropyLoss)
#         x = self.fc2(x)
#         return x

class SimpleCNN(nn.Module): # CNN модель ожидает входные данные в формате (batch_size, channels, height, width), где batch_size - размер батча, channels - количество каналов (1 для ч/б изображений), height и width - высота и ширина изображения соответственно.
    def __init__(self, input_dim=1568, num_hidden=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # Сверточный слой с 1 входным каналом (ч/б изображение), 16 выходными каналами(количество карт признаков), ядром свертки 3x3 и паддингом 1. Этот слой извлекает карту признаков из входного изображения.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Слой подвыборки (пулинга) с ядром 2x2 и шагом 2, который уменьшает размер изображения в 2 раза
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(), # Выпрямление тензора в вектор размера (batch, 1568) перед подачей в полносвязный слой
            nn.Linear(input_dim, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(num_hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)