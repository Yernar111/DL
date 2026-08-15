import torch
from PIL import Image
from model import SimpleCNN

from torchvision import datasets
import torchvision.transforms.v2 as transforms

def predict(dataset, index):
    model = SimpleCNN()
    model.load_state_dict(torch.load("models/best_mnist_model.pth"))
    model.eval()

    image, label = dataset[index]

    image = image.unsqueeze(0) # Добавляем размерность батча (1, 1, 28, 28), так как CNN ожидает входные данные в формате (batch_size, channels, height, width)

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1) # Находим индекс класса с наибольшей вероятностью

    return predicted.item(), label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.MNIST(
        root="dataset",
        train=False,
        download=True,
        transform=transform
    )
    prediction, true_label = predict(test_dataset, index=15) # index=15 - выбираем 15-й элемент из тестового датасета для предсказания

    print(f"True label: {true_label}")
    print(f"Prediction: {prediction}")



