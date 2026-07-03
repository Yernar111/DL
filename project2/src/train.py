import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import DigitNN

import matplotlib.pyplot as plt


def train():
    train_loader, val_loader = get_dataloaders()
    # model = DigitNN()
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(256, 10)
    )
    criterion = nn.CrossEntropyLoss() # Функция потерь для многоклассовой классификации
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # weight_decay - коэффициент регуляризации L2, который помогает предотвратить переобучение модели путем добавления штрафа к слишком большим весам модели.

    loss_ = []
    loss_val = []
    
    epochs = 15
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(
                    images.size(0),
                    28 * 28
                )
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss:.4f}")

        loss_.append(running_loss/len(train_loader.dataset))
        loss_val.append(val_loss/len(val_loader.dataset))

    torch.save(
        model.state_dict(), # Сохраняем только веса модели, а не всю модель целиком. Это позволяет загружать веса в другую модель с той же архитектурой без необходимости сохранять весь объект модели.
        "models/mnist_model1.pth" # Путь для сохранения модели
    )

    print("Model saved") 

    torch.save(
        loss_, # Сохраняем список потерь за эпохи в файл. Это может быть полезно для анализа процесса обучения модели.
        "models/loss1.pth" # Путь для сохранения списка потерь
    )

    torch.save(
        loss_val, # Сохраняем список потерь на валидации за эпохи в файл. Это может быть полезно для анализа процесса обучения модели.
        "models/loss_val1.pth" # Путь для сохранения списка потерь на валидации
    )

    plt.plot(loss_, label="Training Loss")
    plt.plot(loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig("loss_plot1.png") # Сохраняем график потерь в файл.

    plt.show()

if __name__ == "__main__":
    train()