import torch
from PIL import Image
from model import SimpleCNN

from torchvision import datasets
import torchvision.transforms.v2 as transforms

from dataset import get_dataloaders

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
    test_loader = get_dataloaders(batch_size=1, is_train=False)
    
    prediction, true_label = predict(test_loader.dataset, index=45) # .dataset

    print(f"True label: {true_label}")
    print(f"Prediction: {prediction}")



