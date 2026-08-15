import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import torch.utils.data as data

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToImage(), # преобразуем изображение в тензор. В отличие от ToTensor, ToImage не нормализует изображение в диапазон [0, 1], то есть оставляет каждый элемент тензора в диапазоне [0, 255]
        transforms.ToDtype(torch.float32, scale=True), # преобразуем тензор в тип данных float32, чтобы модель могла работать с ним. scale=True нужен для автоматического масштабирования [0,1] путем деления каждого пикселя на 255
        transforms.Normalize(mean=[0.5], std=[0.5]) # Для того чтобы стандартизировать распределение(из диапазона [0,1] в [-1,1]). Это ускоряет обучение и улучшает работу функции активации
    ])

    train_dataset = datasets.MNIST(
        root="dataset",
        train=True,
        download=True,
        transform=transform # Трансформы применяются только при использовании DataLoader
    )

    train_, val_ = data.random_split(train_dataset, [0.7, 0.3])


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