from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.utils.data as data

def get_dataloaders(batch_size=64):
    train_dataset = datasets.MNIST( # метод для загрузки датасета
        root="dataset",
        train=True,
        download=True,
        transform=ToTensor() # преобразуем изображения в тензоры размера (1, 28, 28) где 1 - количество каналов (черно-белое изображение), 28x28 - размер изображения
    )

    # test_dataset = datasets.MNIST(
    #     root="dataset",
    #     train=False,
    #     download=True,
    #     transform=ToTensor()
    # )

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # test_loader = data.DataLoader(
    #     dataset=test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False
    # )

    # return train_loader, test_loader
    return train_loader