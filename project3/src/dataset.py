import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import torch.utils.data as data

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32)
    ])

    train_dataset = datasets.MNIST( # метод для загрузки датасета
        root="dataset",
        train=True,
        download=True,
        transform=transform
    )

    train_, val_ = data.random_split(train_dataset, [0.7, 0.3]) # разделяем датасет на обучающую и валидационную выборки


    train_loader = data.DataLoader(
        dataset=train_,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = data.DataLoader(
        dataset=val_,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader