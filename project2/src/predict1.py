import torch
from torchvision import transforms
from PIL import Image

from model import DigitNN

model = DigitNN()
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.view(1, 28 * 28)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()


if __name__ == "__main__":
    image_path = "dataset/test/test4.png"
    prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")
