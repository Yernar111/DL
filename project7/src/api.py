from fastapi import FastAPI, UploadFile, File, HTTPException
import torch 
from PIL import Image
import io

from torchvision.models import resnet18
from torchvision.transforms import v2

import json

from contextlib import asynccontextmanager
import os
from fastapi.concurrency import run_in_threadpool

MODEL_PATH = "../models/best_model.pth"
CLASSES_PATH = "../models/class_to_idx.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = v2.Compose([
    v2.Resize((224,224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, classes_dict
    
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"File {CLASSES_PATH} not found!")
        
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        raw_classes = json.load(f)
        classes_dict = {idx: name for name, idx in raw_classes.items()}
        
    print(f"Loaded {len(classes_dict)} classes from {CLASSES_PATH}")
    
    num_classes = len(classes_dict)
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model weights are loaded from {MODEL_PATH}")
    else:
        print("Weights not found!")
        
    model.to(device)
    model.eval()
    
    yield
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

def process_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        probs, indices = torch.topk(probabilities, k=3, dim=1)
        
    top_probs = probs[0].tolist()
    top_indices = indices[0].tolist()
    
    results = [
        {
            "class_id": idx,
            "class_name": classes_dict.get(idx, "Unknown"),
            "confidence": round(prob, 4)
        }
        for idx, prob in zip(top_indices, top_probs)
    ]
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image!")
        
    image_bytes = await file.read()
    predictions = await run_in_threadpool(process_image, image_bytes)
    
    return {
        "filename": file.filename,
        "predictions": predictions
    }
