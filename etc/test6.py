import os
import json
from PIL import Image # Библиотека PIL 

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import ImageFolder


class RavelTransform(nn.Module):
    def forward(self, item):
        return item.ravel()


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

transforms = tfs.Compose([
    tfs.ToImage(), 
    tfs.Grayscale(),
    tfs.ToDtype(torch.float32, scale=True),
    RavelTransform(), # Вытягивает все пиксели в один плоский массив
])
d_train = ImageFolder("dataset/train", transform=transforms) # ImageFolder - класс для загрузки изображении из папки, где каждая подпапка соответствует классу. В данном случае, мы загружаем изображения из папки "dataset/train". Трансформы применяются только при использовании DataLoader.
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

# print(d_train.class_to_idx) # class_to_idx - словарь, который отображает имена классов в индексы. В данном случае, он будет выглядеть как {'0': 0, '1': 1, ..., '9': 9}, где ключи - это имена классов (строки), а значения - их индексы (целые числа).
# print(d_train.classes) # classes - список имен классов. В данном случае, он будет выглядеть как ['0', '1', ..., '9'].
# print(d_train.imgs[0]) # imgs - список кортежей (путь к изображению, индекс класса). В данном случае, каждый элемент списка будет выглядеть как ('dataset/train/0/img_1.png', 0), где первый элемент - путь к изображению, а второй элемент - индекс класса.
# print(d_train[0]) # d_train[0] - возвращает кортеж (изображение, индекс класса) для первого изображения в датасете. В данном случае, он будет выглядеть как (tensor([...]), 0), где первый элемент - это тензор изображения, а второй элемент - индекс класса.
# print(len(d_train)) # len(d_train) - возвращает количество изображений в датасете. В данном случае, он будет равен 60000, так как в датасете MNIST 60000 изображений.

optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()
epochs = 2
model.train()

for _e in range(epochs):
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

d_test = ImageFolder("dataset/test", transform=transforms)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

Q = 0

# тестирование обученной НС
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        Q += torch.sum(p == y_test).item()

Q /= len(d_test)
print(Q)