import torch
from torchvision.models import resnet18

from dataset import get_dataloaders

device = torch.device("cpu")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 100)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

def predict(dataset, index):
    image, label = dataset[index]
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1).item()

    class_names = dataset.classes

    return class_names[predicted], class_names[label]


if __name__ == "__main__":
    test_loader = get_dataloaders(batch_size=1, is_train=False)
    
    prediction, true_label = predict(test_loader.dataset, index=3)

    print(f"True label: {true_label}")
    print(f"Prediction: {prediction}")



