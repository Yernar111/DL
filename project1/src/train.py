import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import DigitNN


def train():
    train_loader = get_dataloaders()
    model = DigitNN()
    criterion = nn.CrossEntropyLoss() # Функция потерь для многоклассовой классификации
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # epochs = 10
    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0 # Счетчик для накопления потерь за эпоху
        for images, labels in train_loader:

            # MNIST:
            # [64,1,28,28]
            # ->
            # [64,784]

            images = images.view( # Преобразуем тензор изображений в одномерный вектор размером (64, 784), то есть (размер батча, количество пикселей в изображении)
                images.size(0),
                28 * 28
            )

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # loss.item() возвращает скалярное значение потерь для текущего батча, которое мы добавляем к running_loss для накопления общей потери за эпоху

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {running_loss:.4f}"
        )

    torch.save(
        model.state_dict(), # Сохраняем только веса модели, а не всю модель целиком. Это позволяет загружать веса в другую модель с той же архитектурой без необходимости сохранять весь объект модели.
        "models/mnist_model.pth" # Путь для сохранения модели
    )

    print("Model saved")

if __name__ == "__main__":
    train()