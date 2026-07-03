from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms

import torch.utils.data as data

def get_dataloaders(batch_size=64):
    train_dataset = datasets.MNIST( # метод для загрузки датасета
        root="dataset",
        train=True,
        download=True,
        transform=ToTensor() # преобразуем изображения в тензоры размера (1, 28, 28) где 1 - количество каналов (черно-белое изображение), 28x28 - размер изображения
    )

    train_, val_ = data.random_split(train_dataset, [0.7, 0.3]) # разделяем датасет на обучающую и валидационную выборки в соотношении 5:1

    # test_dataset = datasets.MNIST(
    #     root="dataset",
    #     train=False,
    #     download=True,
    #     transform=ToTensor()
    # )

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

    # test_loader = data.DataLoader(
    #     dataset=test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False
    # )

    
    # return train_loader, test_loader
    return train_loader, val_loader