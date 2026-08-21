import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

class CustomTableDataset(Dataset):
    def __init__(self, features_matrix, targets_vector):
        # Сохраняем данные (в реальности здесь может быть чтение CSV через pandas)
        self.features = torch.tensor(features_matrix, dtype=torch.float32)
        self.targets = torch.tensor(targets_vector, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        # Достаем i-ю строку и i-й ответ
        single_x = self.features[idx]
        single_y = self.targets[idx]
        return single_x, single_y

raw_X = np.random.randn(1000, 4)
raw_y = np.random.randint(0, 2, size=(1000, 1))

my_dataset = CustomTableDataset(raw_X, raw_y)

my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True, num_workers=2)

# 4. Цикл обучения
epochs = 5
for epoch in range(epochs):
    print(f"\n--- Эпоха {epoch+1} ---")
    
    # DataLoader выдает нам данные итеративно батчами
    for batch_idx, (batch_X, batch_y) in enumerate(my_dataloader):
        
        # На этой итерации batch_X имеет форму [32, 4], а batch_y — [32, 1]
        # (На самой последней итерации размер может быть меньше 32, если данные закончились)
        
        # Здесь идет ваш стандартный код:
        # predictions = model(batch_X)
        # loss = criterion(predictions, batch_y)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        
        if batch_idx % 10 == 0:
            print(f"Батч {batch_idx}: Данные улетели в модель. Форма тензора: {batch_X.shape}")