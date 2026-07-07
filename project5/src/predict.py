import torch
from PIL import Image
from torchvision.models import resnet18

from torchvision import datasets
import torchvision.transforms.v2 as transforms

from dataset import get_dataloaders

device = torch.device("cpu")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 53)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device)) # map_location=device ensures that the model is loaded on the CPU, even if it was trained on a GPU
model.to(device)
model.eval()

def predict(dataset, index):

    image, label = dataset[index]

    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1).item() # item() для индекса класса с наибольшей вероятностью

    class_names = dataset.classes

    return class_names[predicted], class_names[label]


if __name__ == "__main__":
    test_loader = get_dataloaders(batch_size=1, is_train=False)
    
    prediction, true_label = predict(test_loader.dataset, index=3)

    print(f"True label: {true_label}")
    print(f"Prediction: {prediction}")



