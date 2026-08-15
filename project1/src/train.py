import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import DigitNN


def train():
    train_loader = get_dataloaders()
    model = DigitNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    # epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0 # Счетчик для накопления потерь за эпоху
        for images, labels in train_loader: # images хранят батч изображений в виде тензора размера [64, 1, 28, 28], а labels хранят батч меток в виде тензора размера [64]

            # # Преобразуем форму, так как модель ожидает входной тензор размером (batch_size, input_dim)
            # [64,1,28,28] -> [64,784]

            images = images.view(images.size(0), 28 * 28) # Преобразуем тензор изображений в одномерный вектор размером (64, 784), то есть (размер батча, количество пикселей в одном изображении).

            outputs = model(images) # Модель возвращает предсказания в виде тензора размера (размер батча, output_dim), где  output_dim=10 - количество классов (цифр от 0 до 9).

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # loss.item() возвращает скалярное(числовое) значение потерь для текущего батча, которое мы добавляем к running_loss для накопления общей потери за эпоху

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {running_loss:.4f}"
        )

    torch.save(model.state_dict(), "models/mnist_model.pth") # Сохраняем только веса модели в определенный путь, а не всю модель целиком. Это позволяет загружать веса в другую модель с той же архитектурой без необходимости сохранять весь объект модели.

    print("Model saved")

if __name__ == "__main__":
    train()