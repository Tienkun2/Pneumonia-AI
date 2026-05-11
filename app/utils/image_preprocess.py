import torch
from torchvision import transforms
from PIL import Image
import io
from app.core.config import settings

def get_image_transform():
    """Returns the transformation pipeline for the vision model."""
    return transforms.Compose([
        transforms.Resize(settings.IMAGE_SIZE),
        transforms.CenterCrop(settings.CENTER_CROP),
        transforms.ToTensor(),
        transforms.Normalize(settings.NORM_MEAN, settings.NORM_STD)
    ])

def preprocess_image(image_bytes: bytes) -> tuple:
    """
    Processes raw image bytes into:
    1. A PyTorch tensor for inference.
    2. A cropped PIL image for heatmap visualization (matching AI's view).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Standard AI Preprocessing
    resize = transforms.Resize(settings.IMAGE_SIZE)
    crop = transforms.CenterCrop(settings.CENTER_CROP)
    normalize = transforms.Normalize(settings.NORM_MEAN, settings.NORM_STD)
    
    # 1. Prepare for AI
    img_resized = resize(image)
    img_cropped = crop(img_resized)
    input_tensor = transforms.ToTensor()(img_cropped).unsqueeze(0)
    input_tensor = normalize(input_tensor)
    
    return input_tensor, img_cropped
