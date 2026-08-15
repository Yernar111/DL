import torch.nn as nn

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 100)