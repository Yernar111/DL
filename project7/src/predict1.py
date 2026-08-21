import torch
from torchvision.transforms import v2
from PIL import Image

from torchvision.models import resnet18

import json
import os

from config import (
    NUM_CLASSES,
    MODEL_PATH,
    CLASS_MAPPING_PATH,
    IMG_PATH,
    IMAGE_SIZE
)

device = torch.device("cpu")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def load_class_mapping(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    return idx_to_class

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted = torch.softmax(output, dim=1)
        probability, predicted = torch.topk(predicted, k=3, dim=1)

    top_probs = probability[0].tolist()
    top_indices = predicted[0].tolist()

    return top_indices, top_probs


if __name__ == "__main__":
    image_path = os.path.join(IMG_PATH, "image8.png")
    idx_to_class = load_class_mapping(CLASS_MAPPING_PATH)
    prediction, probability = predict(image_path)
    for idx, prob in zip(prediction, probability):
        print("class:", idx_to_class[idx], "with confidence:", prob)
