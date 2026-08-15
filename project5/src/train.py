import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import model

import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    train_loader, val_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Оптимизатор обновляет только последний слой model.fc.parameters()

    loss_train = []
    loss_val = []
    
    epochs = 15
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {running_loss:.4f}"
        )

        epoch_loss = running_loss / len(train_loader)

        loss_train.append(epoch_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss:.4f}")

        epoch_val_loss = val_loss / len(val_loader)

        loss_val.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")
        else:
            patience_counter += 1
            print(f"-> Validation Loss не улучшился. Ожидание: {patience_counter}/{patience}")
    
            if patience_counter >= patience:
                print(f" Early Stopping! Обучение остановлено на эпохе {epoch+1}.")
                break

    print("Model saved")

    torch.save(
        loss_train,
        "models/loss1.pth"
    )

    torch.save(
        loss_val,
        "models/loss_val1.pth"
    )

    plt.plot(loss_train, label="Training Loss")
    plt.plot(loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig("loss_plot1.png")

    plt.show()

if __name__ == "__main__":
    train()