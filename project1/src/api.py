from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
from torchvision.transforms import ToTensor

from model import DigitNN

app = FastAPI()

# Загружаем архитектуру и веса из папки models
model = DigitNN()
model.load_state_dict(torch.load("../models/mnist_model.pth"))
model.eval() # Переводим в режим предсказания

@app.post("/predict-digit")
async def predict_digit(file: UploadFile = File(...)):
    # 1. Читаем байты и открываем как картинку PIL
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('L') # Сразу в ч/б (1 канал)
    
    # 2. Приводим к размеру MNIST
    image = image.resize((28, 28))
    
    # 3. Превращаем в тензор (нормализация до [0.0, 1.0] уже внутри ToTensor)
    tensor = ToTensor()(image) # Размерность: [1, 28, 28]
    
    # 4. Выпрямляем в 784 и добавляем размерность батча (batch_size=1) -> [1, 784]
    tensor = tensor.view(1, -1) 
    
    # 5. Делаем предсказание без расчета градиентов
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1) # Применяем softmax, чтобы получить вероятности для каждого класса (цифры от 0 до 9)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    return {
        "predicted_digit": predicted_class,
        "confidence": round(probabilities[0][predicted_class].item(), 4)
    }
