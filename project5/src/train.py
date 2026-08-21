import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import model

import matplotlib.pyplot as plt

import os
import json

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    train_loader, val_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Оптимизатор обновляет только последний слой model.fc.parameters()

    acc_metric = MulticlassAccuracy(num_classes=53).to(device)
    f1_metric = MulticlassF1Score(num_classes=53, average='macro').to(device)

    loss_train = []
    loss_val = []

    acc_val = []
    f1_val = []
    
    epochs = 15
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
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

        print(f"Mean loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                acc_metric.update(outputs, labels)
                f1_metric.update(outputs, labels)

        print(f"Mean validation loss: {val_loss / len(val_loader):.4f}")

        val_acc = acc_metric.compute().item()
        val_f1 = f1_metric.compute().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)

        loss_train.append(epoch_loss)
        loss_val.append(epoch_val_loss)

        acc_val.append(val_acc)
        f1_val.append(val_f1)

        print(f"Эпоха завершена | Accuracy: {val_acc:.4f} | F1-Score: {val_f1:.4f}")
        acc_metric.reset()
        f1_metric.reset()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")
        else:
            patience_counter += 1
            print(f"-> Validation Loss не улучшился. Ожидание: {patience_counter}/{patience}")
    
            if patience_counter >= patience:
                print(f" Early Stopping! Обучение остановлено на эпохе {epoch+1}.")
                break

    print("Model saved")

    loss_history = {"train_loss": loss_train, "val_loss": loss_val}
    with open(os.path.join("models", "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)

    metrics_history = {"val_accuracy": acc_val, "val_f1": f1_val}
    with open(os.path.join("models", "metrics_history.json"), "w") as f:
            json.dump(metrics_history, f, indent=2)

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