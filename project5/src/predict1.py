import torch
from torchvision.transforms import v2
from PIL import Image

from torchvision.models import resnet18

from dataset import get_dataloaders

device = torch.device("cpu")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 53)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

def predict(image_path, card_classes):
    image = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.Resize((128,128)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1)
        idx = predicted.item()

    return card_classes[idx]


if __name__ == "__main__":
    image_path = "data/check/image1.png"
    card_classes = get_dataloaders(batch_size=1, is_train=False).dataset.classes
    prediction = predict(image_path, card_classes)
    print(f"Predicted: {prediction}")
