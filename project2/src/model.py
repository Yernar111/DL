import torch.nn as nn

# class DigitNN(nn.Module):
#     def __init__(self, input_dim=784, num_hidden=256, output_dim=10):
#         super().__init__()
#         self.layer1 = nn.Linear(input_dim, num_hidden)
#         self.layer2 = nn.Linear(num_hidden, output_dim)

#         self.dropout = nn.Dropout(p=0.25) # Dropout слой с вероятностью 0.25, который случайным образом "выключает" 25% нейронов во время обучения, что помогает предотвратить переобучение модели.

#     def forward(self, x):
#         x = self.layer1(x)
#         x = nn.functional.relu(x)
#         x = self.dropout(x)
#         x = self.layer2(x)
#         return x

class DigitNN(nn.Module):
    def __init__(self, input_dim=784, num_hidden=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(num_hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)