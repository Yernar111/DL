import torch.nn as nn

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT) # Веса модели ResNet18, предобученные на ImageNet

for param in model.parameters():
    param.requires_grad = False  # Замораживаем все слои(веса) модели. Это значит, что во время обучения веса этих слоев не будут обновляться. Мы будем обучать только последний слой модели, который мы заменим на новый слой с 2 выходами (для двух классов).

model.fc = nn.Linear(model.fc.in_features, 53)  # В ResNet последний слой называется fc (fully connected layer). Заменяем последний слой на новый с 2 выходами (для двух классов)