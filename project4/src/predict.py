import torch
from PIL import Image
from model import SimpleCNN

from dataset import get_dataloaders

def predict(dataset, index):
    model = SimpleCNN()
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()

    image, label = dataset[index] # dataset[0] возвращает кортеж (изображение, индекс класса) для первого изображения в датасете. В данном случае, он будет выглядеть как (tensor([...]), 0), где первый элемент - это тензор изображения, а второй элемент - индекс класса.

    image = image.unsqueeze(0) # Добавляем размерность батча (1, 1, 28, 28), так как CNN ожидает входные данные в формате (batch_size, channels, height, width)

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1) # Находим индекс класса с наибольшей вероятностью

    return predicted.item(), label


if __name__ == "__main__":
    test_loader = get_dataloaders(batch_size=1, is_train=False)
    
    prediction, true_label = predict(test_loader.dataset, index=45) # test_loader.dataset нужно, чтобы получить доступ к самому датасету, а не к DataLoader. DataLoader используется для итерации по данным в батчах, но для предсказания нам нужен конкретный элемент датасета.

    print(f"True label: {true_label}")
    print(f"Prediction: {prediction}")



