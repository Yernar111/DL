import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # нужно для отображения прогресса обучения в консоли


class DigitDataset(data.Dataset): # Dataset - базовый класс для всех датасетов в PyTorch. Он нужен для того, чтобы можно было итерироваться по датасету батчами. Для этого нужно реализовать методы __getitem__ и __len__.
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(self.path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(path, "format.json"), "r") as fp: # Читаем format.json который содержит словарь, где ключи — имена папок с цифрами, а значения — их числовые обозначения
            self.format = json.load(fp) # Преобразуем JSON в словарь Python. Например, {"zero": 0, "one": 1, "two": 2, ...}

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items(): # Проходимся по словарю
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path) # Получаем список файлов в папке с цифрой
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files)) # Создаем список кортежей (путь_к_файлу, числовое_обозначение_цифры) и добавляем его в self.files

    def __getitem__(self, item):
        path_file, target = self.files[item] # Получаем путь к файлу и числовое обозначение цифры из списка файлов
        t = self.targets[target] # Получаем one-hot вектор для цифры. Например, если target = 3, то t = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img).ravel().float() / 255.0 # Применяем трансформацию к изображению, преобразуем его в одномерный тензор и нормализуем значения пикселей в диапазоне [0, 1]

        return img, t

    def __len__(self):
        return self.length


class DigitNN(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x

model = DigitNN(28 * 28, 32, 10)

# Чтобы модель могла работать с изображениями, нужно преобразовать их в одномерный тензор.
to_tensor = tfs.ToImage()  # PILToTensor. ToImage() преобразует изображение в тензор (канал, высота, ширина). В отличии от ToTensor(), ToImage() не нормализует значения пикселей в диапазоне [0, 1], а оставляет их в диапазоне [0, 255]. Поэтому мы делим на 255.0 в __getitem__ методе класса DigitDataset, чтобы нормализовать значения пикселей.
d_train = DigitDataset("dataset", transform=to_tensor)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True) # DataLoader - класс, который позволяет итерироваться по датасету батчами. batch_size=32 означает, что на каждой итерации мы будем получать 32 изображения. shuffle=True означает, что порядок изображений будет перемешан.

optimizer = optim.Adam(params=model.parameters(), lr=0.01) # Чем больше размер батча, тем больше learning rate должен быть
loss_function = nn.CrossEntropyLoss() # Функция потерь для многоклассовой классификации. Она принимает на вход предсказания модели и правильные ответы, и возвращает значение ошибки. CrossEntropyLoss() объединяет в себе nn.LogSoftmax() и nn.NLLLoss(). Поэтому на выходе модели не нужно применять softmax, так как CrossEntropyLoss() делает это автоматически.
# nn.BCEWithLogitsLoss() # Функция потерь для бинарной классификации 
epochs = 2
model.train()

for _e in range(epochs): # За одну эпоху количество итерации будет равно количеству объектов в датасете деленному на размер батча. За одну итерацию происходит одно обновление весов модели.
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")


d_test = DigitDataset("dataset", train=False, transform=to_tensor)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)
Q = 0

model.eval() # Переводим модель в режим тестирования. В этом режиме отключаются такие вещи, как Dropout и BatchNorm, которые ведут себя по-разному во время обучения и оценки. Это важно, чтобы получить корректные предсказания на тестовых данных.

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1) # torch.argmax() возвращает индексы максимальных значений вдоль указанной размерности(каждой строки матрицы предсказании p). В данном случае, мы получаем индексы классов с наибольшей вероятностью для каждого изображения в батче.
        y = torch.argmax(y_test, dim=1) # 
        Q += torch.sum(p == y).item() # torch.sum(p == y) возвращает количество правильных предсказаний в батче в виде тензора, item() возвращает значение тензора в виде числа. Мы суммируем это количество по всем батчам, чтобы получить общее количество правильных предсказаний на тестовых данных.

Q /= len(d_test) # Получаем долю правильных предсказаний(то есть accuracy)
print(Q)