import torch
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image

from model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("models/best_mnist_model.pth"))
model.eval()

def predict(image_path):
    image = Image.open(image_path)
    transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()


if __name__ == "__main__":
    image_path = "dataset/check/Y71.jpg"
    prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")
