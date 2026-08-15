import torch
from torchvision import transforms
import torchvision.transforms.v2 as transforms
from PIL import Image

from model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("models/best_mnist_model.pth"))
model.eval()

def predict(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Grayscale(), # Преобразуем изображение в оттенки серого. То есть количество каналов будет равно 1
        transforms.Resize((28, 28)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32)
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()


if __name__ == "__main__":
    image_path = "dataset/test/test6.png"
    prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")
