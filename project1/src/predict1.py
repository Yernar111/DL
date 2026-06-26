import torch
from torchvision import transforms
from PIL import Image

from dataset import get_dataloaders
from model import DigitNN

def predict(image_path):
    model = DigitNN()
    model.load_state_dict(
        torch.load("models/mnist_model.pth")
    )
    model.eval()

    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()


if __name__ == "__main__":
    image_path = "dataset/test/0/1.png"  # Путь к изображению для предсказания
    prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")



