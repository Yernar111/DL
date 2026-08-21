# Изображения разделены на train, test, val и на подпапки с классами
import torch
from torchvision import datasets
from torchvision.transforms import v2

import torch.utils.data as data

# Для train
transform1 = v2.Compose([
    v2.Resize((128,128)), # Для модели ResNet18 подходят изображения 224х224, но можно подавать меньшие размеры для более быстрого обучения или большие размеры для точности(нужно чтобы размер делился на 32)
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(10),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Значения для CNN моделей обученной на ImageNet
])

# Для val, test
transform2 = v2.Compose([
    v2.Resize((128,128)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders(batch_size=64, is_train=True):
    if is_train:
        train_dataset = datasets.ImageFolder(
            root="data/train_test_val/train",
            transform=transform1
        )
        val_dataset = datasets.ImageFolder(
            root="data/train_test_val/valid",
            transform=transform2
        )
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, val_loader
    else:
        test_dataset = datasets.ImageFolder(
            root="data/train_test_val/test",
            transform=transform2
        )
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return test_loader