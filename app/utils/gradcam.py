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
    Grad-CAM implementation optimized for PyTorch 2.x and GPU environments.
    Uses a forward hook and a direct tensor backward hook to reliably capture
    activations and gradients on any device/version.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    def _save_activations(self, module, input, output):
        self.activations = output.detach()
        
        # Register a backward hook directly on the output tensor.
        # This is 100% robust across all PyTorch versions and GPU/autocast modes.
        def _save_gradients(grad):
            self.gradients = grad.detach()
            
        output.register_hook(_save_gradients)

    def generate(self, input_tensor, img_cropped):
        """Generates heatmap with High-Res sharpening (Power 2)."""
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True

        self.activations = None
        self.gradients = None

        # Register forward hook
        h_f = self.target_layer.register_forward_hook(self._save_activations)

        try:
            self.model.zero_grad()
            
            # Forward pass in standard float32 (disabling autocast to ensure hooks trigger on GPU)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                input_tensor = input_tensor.float()
                output = self.model(input_tensor)
            
            # Backward pass on the scalar mean (robust to any shape)
            output.mean().backward()

            if self.gradients is None or self.activations is None:
                logger.error("GRAD-CAM FAILED: Gradients or Activations not captured.")
                return None
            
            # Process Gradients and Activations
            gradients_np = self.gradients.cpu().data.numpy()[0]
            activations_np = self.activations.cpu().data.numpy()[0]
            
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
                # HIGH-CONTRAST SHARPENING (Power 2 for smoother visualization)
                cam = np.power(cam, 2.0)
                # Noise filtering
                cam = np.where(cam > 0.02, cam, 0)
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
            h_f.remove()

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
