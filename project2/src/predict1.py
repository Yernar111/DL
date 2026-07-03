import torch
from torchvision import transforms
from PIL import Image

from model import DigitNN

model = DigitNN()
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("L") # Открываем изображение и конвертируем в градации серого
    transform = transforms.Compose([ # Определяем последовательность преобразований
        transforms.Resize((28, 28)), # Изменяем размер изображения до 28x28 пикселей
        transforms.ToTensor(), # Преобразуем изображение в тензор
        # transforms.Normalize((0.5,), (0.5,)) # Нормализуем тензор изображения, чтобы значения пикселей находились в диапазоне [-1, 1]
    ])
    # image = transform(image).unsqueeze(0)
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
