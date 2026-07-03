import torch
a = torch.load("models/loss_val1.pth", weights_only=True)
print(a)