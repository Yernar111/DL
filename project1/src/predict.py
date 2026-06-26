import torch
from torchvision import transforms
from PIL import Image

from dataset import get_dataloaders
from model import DigitNN

from torchvision import datasets
from torchvision.transforms import ToTensor

def predict(dataset, index):
    model = DigitNN()
    model.load_state_dict(torch.load("models/mnist_model.pth")) # Загружаем сохраненные веса модели
    model.eval()

    image, label = dataset[index]

    # Преобразуем форму:
    # [1, 28, 28] -> [1, 784]
    image = image.view(1, 28 * 28)

    # Предсказание
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1) # Находим индекс класса с наибольшей вероятностью

    return predicted.item(), label


if __name__ == "__main__":
    test_dataset = datasets.MNIST(
        root="dataset",
        train=False,
        download=True,
        transform=ToTensor()
    )
    prediction, true_label = predict(test_dataset, index=15) # index=15 - выбираем 15-й элемент из тестового датасета для предсказания

    print(f"True label: {true_label}")
    print(f"Prediction: {prediction}")



