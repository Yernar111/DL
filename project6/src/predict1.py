import torch
from torchvision.transforms import v2
from PIL import Image

from torchvision.models import resnet18

from dataset import get_dataloaders

device = torch.device("cpu")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 100)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

def predict(image_path, classes):
    image = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    # with torch.no_grad():
    #     output = model(image)
    #     # predicted = torch.argmax(output, dim=1)
    #     _, predicted = torch.max(output.data, 1)
    #     idx = predicted.item()

    # return classes[idx], _.item()

    with torch.no_grad():
        output = model(image)
        predicted = torch.softmax(output, dim=1)
        probability, predicted = torch.topk(predicted, k=3, dim=1)

    top_probs = probability[0].tolist()
    top_indices = predicted[0].tolist()

    return top_indices, top_probs


if __name__ == "__main__":
    image_path = "data1/image2.png"
    classes = get_dataloaders(batch_size=1, is_train=False).dataset.classes
    prediction, probability = predict(image_path, classes)
    # print(f"Predicted: {prediction}")
    # print(f"Probability: {probability}")
    for idx, prob in zip(prediction, probability):
        print("class:", classes[idx], "with confidence:", prob)
