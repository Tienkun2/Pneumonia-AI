import torch
from torchvision import transforms
from PIL import Image
import io
from app.core.config import settings

def get_image_transform():
    """Returns the transformation pipeline for the vision model."""
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def preprocess_image(image_bytes: bytes) -> tuple:
    """
    Processes raw image bytes into:
    1. A PyTorch tensor for inference.
    2. A resized PIL image for heatmap visualization (matching training size).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Matching the exact training transforms (Resize 448x448, no CenterCrop)
    resize = transforms.Resize((448, 448))
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    img_resized = resize(image)
    input_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)
    input_tensor = normalize(input_tensor)
    
    return input_tensor, img_resized
