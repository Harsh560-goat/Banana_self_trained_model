import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os 
from torchvision import datasets, transforms
from PIL import Image

# Defining model architecture
class BananaModel(nn.Module):
    def __init__(self, num_classes=4):
        super(BananaModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.ANN = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*56*56, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.feature(x)
        x = self.ANN(x)
        return x

# Checking if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Loading the weights from already trained model
model = BananaModel()
model = model.to(device)
model.load_state_dict(torch.load(os.path.join("banana_model2.pth"), map_location=device))
model.eval()

# Transform function for the image 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Banana classes
class_names = ['overripe', 'ripe', 'rotten', 'unripe']

# For Single Image Prediction
def predict_single(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)           # torch.max(output, 1) returns two values: the maximum values and their indices
                                                      # _ ignores the maximum values (we don't need them)
                                                      # predicted stores the index of the class with the highest score (0-3 for the 4 classes)
    return class_names[predicted.item()]

# For Batch Prediction
def predict_batch(image_folder):
    image_dataset = datasets.ImageFolder(image_folder, transform=transform)
    image_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for images, _ in image_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    for prediction in predictions:
        print(class_names[prediction])
