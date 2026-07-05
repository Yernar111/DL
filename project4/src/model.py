import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=25088, num_hidden=512, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            # Вход: [Батч, 3, 224, 224] (3 канала — цветная картинка)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Вход: [Батч, 32, 224, 224]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Вход: [Батч, 32, 112, 112]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Вход: [Батч, 64, 112, 112]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Вход: [Батч, 64, 56, 56]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Вход: [Батч, 128, 56, 56]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Вход: [Батч, 128, 28, 28]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Вход: [Батч, 256, 28, 28]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Вход: [Батч, 256, 14, 14]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # Вход: [Батч, 512, 14, 14]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Вход: [Батч, 512, 7, 7]
            nn.Flatten(),
            nn.Linear(input_dim, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(num_hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)