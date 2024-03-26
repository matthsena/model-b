import cv2
from torchvision import models, transforms
from PIL import Image
import torch

import torch.nn.functional as F

model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load classes only once
with open('./resources/imagenet_classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define transform outside the predict function
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225]),
])

def decode_predictions(preds, top=3):
    probabilities = F.softmax(preds, dim=1)[0]
    _, indices = torch.topk(probabilities, top)
    decoded_preds = [(classes[idx], probabilities[idx].item())
                     for idx in indices]
    return decoded_preds

def predict(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    x = transform(image).unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        preds = model(x)

    return decode_predictions(preds, top=5)
