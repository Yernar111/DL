import os
import pandas as pd
import torch
from torch.utils import data
from PIL import Image
from torchvision.transforms import v2

import json

from config import (
    BATCH_SIZE,
    TRAIN_DIR,
    TRAIN_CSV,
    CLASS_MAPPING_PATH,
    TRAIN_SIZE,
    IMAGE_SIZE,
    RANDOM_SEED
)

transform1 = v2.Compose([
    v2.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(data.Dataset):
    def __init__(self, csv_path, img_dir, transform=None, class_to_idx=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_id)
        
        image = Image.open(img_path).convert("RGB")
        
        label_name = self.df.iloc[idx, 1]

        label = self.class_to_idx[label_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def create_class_mapping(csv_path, json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    df = pd.read_csv(csv_path)

    classes = sorted(df.iloc[:, 1].unique())
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=4)

    return class_to_idx

def get_dataloaders(batch_size=BATCH_SIZE, is_train=True):
    if is_train:
        class_to_idx = create_class_mapping(TRAIN_CSV, CLASS_MAPPING_PATH)
        train_dataset = CustomImageDataset(
            csv_path=TRAIN_CSV,
            img_dir=TRAIN_DIR,
            transform=transform1,
            class_to_idx=class_to_idx
        )

        train_size = int(TRAIN_SIZE * len(train_dataset))
        val_size = len(train_dataset) - train_size

        generator=torch.Generator().manual_seed(RANDOM_SEED)

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

