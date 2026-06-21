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
    Grad-CAM for PyTorch 2.x.  Forward hook + tensor backward hook for reliable
    activation/gradient capture on any device or autocast mode.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

        def _save_gradients(grad):
            self.gradients = grad.detach()

        output.register_hook(_save_gradients)

    def generate(self, input_tensor, img_cropped, lung_mask: np.ndarray = None):
        """
        Generate Grad-CAM heatmap overlay.

        Returns:
            heatmap_b64    – base64 JPEG string or None on error
            error_msg      – error detail string or None on success
            lung_focus_ratio – float in [0,1] if lung_mask provided, else None
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).detach()
        input_tensor.requires_grad = True

        self.activations = None
        self.gradients = None

        h_f = self.target_layer.register_forward_hook(self._save_activations)

        try:
            self.model.zero_grad()

            # Float32 forward — disabling autocast ensures hooks fire on GPU
            device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                input_tensor = input_tensor.float()
                output = self.model(input_tensor)

            output.mean().backward()

            if self.gradients is None or self.activations is None:
                logger.error("GRAD-CAM FAILED: Gradients or Activations not captured.")
                return None, "Gradients or Activations not captured.", None

            gradients_np = self.gradients.cpu().data.numpy()[0]
            activations_np = self.activations.cpu().data.numpy()[0]

            weights = np.mean(gradients_np, axis=(1, 2))
            cam = np.zeros(activations_np.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations_np[i, :, :]

            cam = np.maximum(cam, 0)
            v_max = np.max(cam)
            if v_max <= 0:
                logger.warning("Zero CAM activations.")
                return None, "Zero CAM activations.", None

            cam = cam / v_max
            cam = np.power(cam, 2.0)           # high-contrast sharpening
            cam = np.where(cam > 0.02, cam, 0) # noise filter

            # Resize CAM to output image size
            out_w, out_h = img_cropped.size
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize((out_w, out_h), resample=Image.BICUBIC)
            ).astype(np.float32) / 255.0

            # lung_focus_ratio (spec §5.3): fraction of CAM energy inside lung mask
            lung_focus_ratio = None
            if lung_mask is not None:
                # cam_resized is (H,W); lung_mask is bool (H,W) at same size
                mask_resized = lung_mask
                if mask_resized.shape != cam_resized.shape:
                    from PIL import Image as _PIL
                    mask_resized = np.array(
                        _PIL.fromarray(lung_mask.astype(np.uint8)).resize(
                            (cam_resized.shape[1], cam_resized.shape[0]), resample=Image.NEAREST
                        )
                    ).astype(bool)
                lung_focus_ratio = float(cam_resized[mask_resized].sum() / (cam_resized.sum() + 1e-8))
                # Clear CAM heat outside of the lung mask to prevent upsampling bleed/smear on display
                cam_resized[~mask_resized] = 0

            # Build overlay: Jet colormap via alpha paste
            heatmap_pil = Image.fromarray((cam_resized * 255).astype(np.uint8))
            heatmap_pil = heatmap_pil.filter(ImageFilter.GaussianBlur(radius=4))
            overlay = self._apply_colormap(heatmap_pil)

            alpha = Image.fromarray((cam_resized * 180).astype(np.uint8)).resize(
                img_cropped.size, resample=Image.BICUBIC
            )
            result = img_cropped.copy()
            result.paste(overlay, (0, 0), mask=alpha)

            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode(), None, lung_focus_ratio

        except Exception as e:
            import traceback
            err_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"GRAD-CAM GENERATION ERROR: {err_msg}")
            return None, err_msg, None
        finally:
            h_f.remove()

    def _apply_colormap(self, heatmap_gray: Image.Image) -> Image.Image:
        """Grayscale → Jet-like pseudo-thermal colormap."""
        heatmap_gray = heatmap_gray.convert("L")
        palette = []
        for i in range(256):
            if i < 128:
                palette.extend([0, i * 2, 255 - i * 2])
            else:
                palette.extend([(i - 128) * 2, 255 - (i - 128) * 2, 0])
        heatmap_gray.putpalette(palette)
        return heatmap_gray.convert("RGB")
