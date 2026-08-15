import torch
from torchvision import transforms
from PIL import Image

from model import DigitNN

model = DigitNN()
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("L") # Открываем изображение и конвертируем в градации серого
    transform = transforms.Compose([ # Compose позволяет объединять несколько трансформаций в одну. Определяем последовательность преобразований
        transforms.Resize((28, 28)), # Изменяем размер изображения до 28x28 пикселей
        transforms.ToTensor(), # Преобразуем изображение в тензор
        # transforms.Normalize((0.5,), (0.5,)) # Нормализуем тензор изображения, чтобы значения пикселей находились в диапазоне [-1, 1]
    ])
    # image = transform(image).unsqueeze(0)
    image = transform(image)
    image = image.view(1, 28 * 28)

    # Старый способ(не рекомендуется), используется если надо получить не только предсказание но и уверенность модели в процентах
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1) # Возвращает кортеж из двух тензоров (максимальное значение, ее индекс) вдоль указанной размерности. Здесь мы используем 1, чтобы найти индекс максимального элемента в каждой строке (каждого изображения в батче). Индекс соответствует предсказанному классу (цифре от 0 до 9).

    return _.item(), predicted.item()


if __name__ == "__main__":
    image_path = "dataset/test/test1.png"
    probability, prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")
    print(f"Probability: {probability}")