from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# Load a pretrained YOLOv8 model (nano version for speed)
model = YOLO("yolov8n.pt")

# Load an image (URL or local file)
image_path = "https://ultralytics.com/images/bus.jpg"

# Run object detection
results = model(image_path)

# Show results
results[0].show()

# Optional: Save result image to disk
results[0].save(filename="detected.jpg")

# Optional: Display result image using matplotlib
img = Image.open("detected.jpg")
plt.imshow(img)
plt.axis("off")
plt.title("YOLOv8 Detection")
plt.show()
