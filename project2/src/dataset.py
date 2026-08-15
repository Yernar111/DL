from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms

import torch.utils.data as data

def get_dataloaders(batch_size=64):
    train_dataset = datasets.MNIST(
        root="dataset",
        train=True,
        download=True,
        transform=ToTensor()
    )

    train_, val_ = data.random_split(train_dataset, [0.7, 0.3]) # разделяем датасет на обучающую и валидационную выборки в соотношении 70% и 30%. random_split возвращает два объекта Subset, которые представляют собой подмножества исходного датасета.

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