import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import requests
from io import BytesIO

# Load pretrained ResNet18 with new 'weights' syntax
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Download a working image
url = "https://ultralytics.com/images/zidane.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('RGB')

# Preprocess
preprocess = weights.transforms()
input_tensor = preprocess(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    predicted_idx = output.argmax().item()

# Get labels
labels = weights.meta["categories"]
print("Predicted:", labels[predicted_idx])

