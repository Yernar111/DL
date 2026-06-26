import torch.nn as nn

class DigitNN(nn.Module):
    def __init__(self, input_dim=784, num_hidden=64, output_dim=10):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x