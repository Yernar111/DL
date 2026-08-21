# Изображения разделены на train и test, но отсутствуют подпапки с классами. Прилагается csv файл с изображениями и метками
# Удобно для обучения в google colab
import os
import pandas as pd
import torch
from torch.utils import data
from PIL import Image
from torchvision.transforms import v2

import json

transform1 = v2.Compose([
    v2.Resize((224,224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(data.Dataset):
    def __init__(self, csv_path, img_dir, transform=None, class_to_idx=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        if class_to_idx is None: # Если mapping не передан
            classes = sorted(self.df.iloc[:, 1].unique())

            # Создаем mapping
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)} # метод enumerate() возвращает кортеж для каждого элемента (индекс, элемент)

            # with open("models/class_to_idx.json", "w", encoding="utf-8") as f:
            #         json.dump(class_to_idx, f, ensure_ascii=False, indent=4)
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Берем имя изображения из первой колонки CSV
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_id)
        
        image = Image.open(img_path).convert("RGB")
        
        # Берем метку класса из второй колонки CSV
        label_name = self.df.iloc[idx, 1]

        label = self.class_to_idx[label_name]
        
        # 4. Применяем трансформации
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(batch_size=64, is_train=True):
    if is_train:
        train_dataset = CustomImageDataset(
            csv_path="data/Training_set.csv",
            img_dir="data/train",
            transform=transform1
        )
        class_to_idx = train_dataset.class_to_idx

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        generator=torch.Generator().manual_seed(42)

        train_dataset, val_dataset = data.random_split(
            train_dataset, 
            [train_size, val_size],
            generator=generator
        )
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return train_loader, val_loader

    else:
        test_dataset = CustomImageDataset(
            csv_path="data/Testing_set.csv",
            img_dir="data/test",
            transform=transform1,
            class_to_idx=class_to_idx # Передаем ранее созданный mapping
        )
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return test_loader