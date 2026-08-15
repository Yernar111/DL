# Simple api
from fastapi import FastAPI, UploadFile, File
import torch 
from PIL import Image
import io

from dataset import get_dataloaders
from torchvision.models import resnet18
from torchvision.transforms import v2

classes = get_dataloaders(batch_size=1, is_train=False).dataset.classes
device = torch.device("cpu")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 100)
model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = v2.Compose([
    v2.Resize((224,224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        predicted = torch.softmax(output, dim=1)
        probability, predicted = torch.topk(predicted, k=3, dim=1)
    
    top_probs = probability[0].tolist()
    top_indices = predicted[0].tolist()

    results = [
        {"class": classes[idx], "confidence": round(prob, 4)}
        for idx, prob in zip(top_indices, top_probs)
    ]

    return results
