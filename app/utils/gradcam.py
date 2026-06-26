import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import io
import base64
import logging
import cv2

DISCLAIMER = ("Các chỉ số hình thái là mô tả vùng mô hình tập trung (Grad-CAM), "
              "KHÔNG phải phép đo diện tích tổn thương hay mức độ nặng.")

def describe_cam_foci(cam, lung_mask = None, hot_thr=0.5,
                      min_focus_frac=0.004, diffuse_area_frac=0.25):
    """
    cam        : np.ndarray [H,W], giá trị [0..1] (đã resize về kích thước ảnh).
    lung_mask  : np.ndarray [H,W] bool/0-1, vùng phổi.
    hot_thr    : ngưỡng "nóng" trên CAM để coi là vùng tập trung.
    min_focus_frac : ổ nhỏ hơn tỉ lệ này so với diện phổi -> bỏ (lọc nhiễu).
    diffuse_area_frac : 1 ổ phủ quá tỉ lệ này diện phổi -> coi là 'Lan tỏa'.
    """
    H, W = cam.shape
    if lung_mask is None:
        lung_mask = np.ones_like(cam, dtype=bool)
    m = lung_mask.astype(bool)
    lung_area = int(m.sum()) + 1

    # --- (1) Độ tập trung năng lượng CAM trong phổi (chỉ số tin cậy định vị) ---
    attention_in_lung = float((cam * m).sum() / (cam.sum() + 1e-8))

    # --- (2) Vùng "nóng" trong phổi ---
    hot = ((cam > hot_thr) & m).astype(np.uint8)
    hot_area_frac = float(hot.sum() / lung_area)

    # --- (3) Trục giải phẫu: đường giữa (chia trái/phải) + biên trên/dưới của phổi ---
    cols = np.where(m.any(axis=0))[0]
    rows = np.where(m.any(axis=1))[0]
    midline = (cols.min() + cols.max()) / 2.0 if len(cols) else W / 2.0
    y0, y1 = (rows.min(), rows.max()) if len(rows) else (0, H)

    # --- (4) Đếm ổ bằng connected components ---
    n_lbl, _, stats, cents = cv2.connectedComponentsWithStats(hot, connectivity=8)
    min_px = max(1, int(min_focus_frac * lung_area))
    foci = []
    for i in range(1, n_lbl):                       # bỏ nền (label 0)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        cx, cy = cents[i]
        # QUY ƯỚC PHIM LẬT GƯƠNG: nửa-trái-ảnh = phổi PHẢI của bệnh nhân
        side = "phổi phải" if cx < midline else "phổi trái"
        frac = (cy - y0) / (y1 - y0 + 1e-8)
        zone = "đỉnh/trên" if frac < 1 / 3 else ("giữa" if frac < 2 / 3 else "đáy/dưới")
        foci.append({"side": side, "zone": zone,
                     "area_pct": round(100 * area / lung_area, 1),
                     "cx": float(cx), "cy": float(cy)})
    foci.sort(key=lambda f: -f["area_pct"])          # ổ lớn nhất trước
    n = len(foci)

    # --- (5) PHÂN BỐ: một bên / hai bên ---
    sides = {f["side"] for f in foci}
    if len(sides) >= 2:
        distribution = "Hai bên"
    elif len(sides) == 1:
        s = next(iter(sides))
        distribution = "Một bên (" + ("phải)" if "phải" in s else "trái)")
    else:
        distribution = "Không rõ"

    # --- (6) ĐẶC ĐIỂM: lan tỏa / đa ổ / đơn ổ — NHẤT QUÁN với số ổ ---
    if n == 0:
        characteristic = "Không khu trú rõ"
    elif n == 1 and hot_area_frac > diffuse_area_frac:
        characteristic = "Lan tỏa"
    elif n == 1:
        characteristic = "Đơn ổ"
    else:
        characteristic = f"Đa ổ ({n})"

    # --- (7) VỊ TRÍ: liệt kê ổ (khớp với 'đa ổ', không chốt sai 1 bên) ---
    def cap(s):
        return s[:1].upper() + s[1:]
    if n == 0:
        location = "Không khu trú rõ"
    elif n == 1:
        location = cap(f"{foci[0]['zone']} {foci[0]['side']}")
    else:
        parts = [f"{f['zone']} {f['side']}" for f in foci[:3]]
        extra = "" if n <= 3 else f" (+{n - 3})"
        location = f"{n} ổ: " + "; ".join(parts) + extra

    # --- (8) Câu mô tả TRUNG THỰC (không có 'giai đoạn/nhẹ') ---
    if n == 0:
        desc = ("Grad-CAM không nổi vùng tập trung rõ trong phổi ở ngưỡng hiện tại; "
                "kết luận dựa chủ yếu vào điểm tin cậy tổng thể.")
    else:
        spread = "lan tỏa" if characteristic == "Lan tỏa" else f"phân bố {distribution.lower()}"
        desc = (f"Grad-CAM nổi {n} vùng tập trung ({location.lower()}), {spread}. "
                f"Vùng chú ý mạnh (CAM>{hot_thr}) chiếm ~{round(hot_area_frac*100,1)}% diện phổi. "
                f"{DISCLAIMER}")

    return {
        "location_label": location,            # VỊ TRÍ
        "distribution_label": distribution,    # PHÂN BỐ
        "characteristic_label": characteristic,# ĐẶC ĐIỂM
        "foci_count": n,
        "foci": foci,
        "attention_in_lung_pct": round(attention_in_lung * 100, 1),
        "hot_area_pct": round(hot_area_frac * 100, 1),
        "description": desc,
        "disclaimer": DISCLAIMER,
    }


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
            heatmap_b64      – base64 JPEG string or None on error
            error_msg        – error detail string or None on success
            lung_focus_ratio – float in [0,1] if lung_mask provided, else None
            cam_metrics      – dict with 5 clinical indicators or None
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
                return None, "Gradients or Activations not captured.", None, None

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
                return None, "Zero CAM activations.", None, None

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
            cam_metrics = {
                "location_label": "Không khu trú rõ",
                "distribution_label": "Không rõ",
                "characteristic_label": "Không khu trú rõ",
                "foci_count": 0,
                "foci": [],
                "attention_in_lung_pct": 0.0,
                "hot_area_pct": 0.0,
                "description": "",
                "disclaimer": DISCLAIMER,
            }
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
                
                # Compute clinical metrics before zeroing outside mask using describe_cam_foci
                cam_metrics = describe_cam_foci(cam_resized, mask_resized)
                
                # Clear CAM heat outside of the lung mask to prevent upsampling bleed/smear on display
                cam_resized[~mask_resized] = 0
            else:
                # If lung mask is None, we still compute metrics without lung mask constraints
                cam_metrics = describe_cam_foci(cam_resized, None)

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
            return base64.b64encode(buffered.getvalue()).decode(), None, lung_focus_ratio, cam_metrics

        except Exception as e:
            import traceback
            err_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"GRAD-CAM GENERATION ERROR: {err_msg}")
            return None, err_msg, None, None
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
