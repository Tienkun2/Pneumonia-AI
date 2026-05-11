import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import io
import base64
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GradCAM:
    """
    Grad-CAM implementation optimized for EfficientNet-B4 and Mixed Precision.
    Uses manual forward tracking to avoid hook issues with Autocast.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def _save_gradients(self, module, grad_input, grad_output):
        # Using legacy backward hook as it handles autocast scaled gradients better in some torch versions
        self.gradients = grad_output[0]

    def generate(self, input_tensor, img_cropped):
        """Generates heatmap with High-Res sharpening (Power 4)."""
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True

        # Register hook for gradients
        # Use older register_backward_hook if requested for autocast compatibility
        h_b = self.target_layer.register_backward_hook(self._save_gradients)

        try:
            self.model.zero_grad()
            
            # Forward pass with Autocast
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                x = input_tensor
                activations = None
                
                # Manual iteration through features to capture activations exactly
                for i, layer in enumerate(self.model.features):
                    x = layer(x)
                    if layer == self.target_layer:
                        activations = x.clone()
                
                # Complete the forward pass
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                output = self.model.classifier(x)
            
            # Backward pass
            output.backward()

            if self.gradients is None or activations is None:
                logger.error("GRAD-CAM FAILED: Gradients or Activations not captured.")
                return None
            
            # Process Gradients and Activations
            gradients_np = self.gradients.cpu().data.numpy()[0]
            activations_np = activations.cpu().data.numpy()[0]
            
            # Global Average Pooling for Gradients
            weights = np.mean(gradients_np, axis=(1, 2))
            
            # Compute CAM
            cam = np.zeros(activations_np.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations_np[i, :, :]
            
            # ReLU and Normalization
            cam = np.maximum(cam, 0)
            v_max = np.max(cam)
            if v_max > 0:
                cam = cam / v_max
                # HIGH-CONTRAST SHARPENING (Power 4 for Hilum focus)
                cam = np.power(cam, 4.0)
                # Noise filtering
                cam = np.where(cam > 0.1, cam, 0)
            else:
                logger.warning("Zero CAM activations.")
                return None

            # Render to Image
            heatmap_gray = Image.fromarray((cam * 255).astype(np.uint8)).resize(img_cropped.size, resample=Image.BICUBIC)
            heatmap_gray = heatmap_gray.filter(ImageFilter.GaussianBlur(radius=4))
            
            overlay = self._apply_colormap(heatmap_gray)
            
            # Alpha Masking
            mask_arr = (cam * 180).astype(np.uint8)
            mask = Image.fromarray(mask_arr).resize(img_cropped.size, resample=Image.BICUBIC)
            
            result = img_cropped.copy()
            result.paste(overlay, (0, 0), mask=mask)
            
            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode()

        except Exception as e:
            logger.error(f"GRAD-CAM GENERATION ERROR: {str(e)}")
            return None
        finally:
            h_b.remove()

    def _apply_colormap(self, heatmap_gray):
        """Converts grayscale to a pseudo-thermal (Jet) color map."""
        heatmap_gray = heatmap_gray.convert("L")
        palette = []
        for i in range(256):
            if i < 128:
                palette.extend([0, i * 2, 255 - i * 2])
            else:
                palette.extend([(i - 128) * 2, 255 - (i - 128) * 2, 0])
        
        heatmap_gray.putpalette(palette)
        return heatmap_gray.convert("RGB")
