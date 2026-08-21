import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import model

import matplotlib.pyplot as plt

import os
import json

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from config import (
    NUM_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
    MODEL_PATH,
    LOSS_HISTORY_PATH,
    METRICS_HISTORY_PATH,
    DEVICE
)

def run_epoch(model, loader, criterion, device, optimizer=None, acc_metric=None, f1_metric=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                acc_metric.update(outputs, labels)
                f1_metric.update(outputs, labels)

            total_loss += loss.item()

    if is_train:
        return total_loss / len(loader)
    else:
        val_acc = acc_metric.compute().item()
        val_f1 = f1_metric.compute().item()
        return total_loss / len(loader), val_acc, val_f1

def train():
    device = DEVICE
    model.to(device)
    print(f"Using device: {device}")
    train_loader, val_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(device)

    loss_train = [] 
    loss_val = []

    acc_val = []
    f1_val = []
    
    epochs = NUM_EPOCHS
    best_val_loss = float('inf')
    patience = PATIENCE
    patience_counter = 0
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device, optimizer=None, acc_metric=acc_metric, f1_metric=f1_metric)

        print(f"Mean loss: {train_loss:.4f}")

        print(f"Mean validation loss: {val_loss:.4f}")

        loss_train.append(train_loss)
        loss_val.append(val_loss)

        acc_val.append(val_acc)
        f1_val.append(val_f1)

        print(f"Эпоха завершена | Accuracy: {val_acc:.4f} | F1-Score: {val_f1:.4f}")
        acc_metric.reset()
        f1_metric.reset()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), "models/best_model.pth")
            torch.save(model.state_dict(), MODEL_PATH)
            print("New best model saved")
        else:
            patience_counter += 1
            print(f"-> Validation Loss не улучшился. Ожидание: {patience_counter}/{patience}")
    
            if patience_counter >= patience:
                print(f" Early Stopping! Обучение остановлено на эпохе {epoch+1}.")
                break

    loss_history = {"train_loss": loss_train, "val_loss": loss_val}
    with open(LOSS_HISTORY_PATH, "w") as f:
        json.dump(loss_history, f, indent=2)

    metrics_history = {"val_accuracy": acc_val, "val_f1": f1_val}
    with open(METRICS_HISTORY_PATH, "w") as f:
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