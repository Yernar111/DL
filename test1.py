import torch

x1 = torch.tensor([1, 2, 3])

# print(x1.shape)
# print(x1.dtype)

x2 = torch.tensor([[1, 2], [3, 4]])

# print(x2)

ones = torch.ones(2, 2)         # Матрица 2x2 из единиц
# tensor([[1., 1.],
#         [1., 1.]])

rand = torch.rand(2, 3)         # Случайные числа из равномерного распределения от 0 до 1
# tensor([[0.4273, 0.4753, 0.1979],
#         [0.6967, 0.1719, 0.7664]])

randn = torch.randn(2, 3)       # Случайные числа из нормального распределения (mean=0, std=1)
#
#

x3 = torch.tensor([[1, 2], [3, 4]])

