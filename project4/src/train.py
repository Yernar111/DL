import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import SimpleCNN

import matplotlib.pyplot as plt

def train():
    train_loader, val_loader = get_dataloaders()
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    loss_train = []
    loss_val = []
    
    epochs = 12
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader: # images имеют размерность (batch_size, 1, 28, 28)
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
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss:.4f}")

        loss_train.append(running_loss/len(train_loader))
        loss_val.append(val_loss/len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth") # Сохраняем лучшую версию весов на диск
        else:
            patience_counter += 1
            print(f"-> Validation Loss не улучшился. Ожидание: {patience_counter}/{patience}")
    
            if patience_counter >= patience:
                print(f" Early Stopping! Обучение остановлено на эпохе {epoch+1}.")
                break

    print("Model saved")

    torch.save(
        loss_train, # Сохраняем список потерь за эпохи в файл. Это может быть полезно для анализа процесса обучения модели.
        "models/loss1.pth" # Путь для сохранения списка потерь
    )

    torch.save(
        loss_val, # Сохраняем список потерь на валидации за эпохи в файл. Это может быть полезно для анализа процесса обучения модели.
        "models/loss_val1.pth" # Путь для сохранения списка потерь на валидации
    )

    plt.plot(loss_train, label="Training Loss")
    plt.plot(loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig("loss_plot1.png") # Сохраняем график потерь в файл.

    plt.show()

if __name__ == "__main__":
    train()