import torch
from torchvision import datasets
from torchvision.transforms import v2

import torch.utils.data as data

from torch.utils.data import random_split

def get_dataloaders(batch_size=64, is_train=True):
    transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(
        root="dataset/train_test",
        transform=transform
    )
    # print(dataset.class_to_idx)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator=torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )

    if is_train:
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return train_loader, val_loader

    else:
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return test_loader