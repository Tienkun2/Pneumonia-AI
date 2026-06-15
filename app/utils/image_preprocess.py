import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import logging
import cv2
from app.core.config import settings

logger = logging.getLogger(__name__)


def _keep_two_lungs(binary: np.ndarray) -> np.ndarray:
    """
    Post-process PSPNet/CV2 mask:
      1. Giữ đúng 2 thành phần liên thông lớn nhất (= 2 lá phổi)
      2. Fill holes bên trong mỗi lá phổi
    → loại bỏ mảnh vụn, không còn đen tùm lum bên trong phổi.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return binary

    # Sắp xếp theo diện tích (bỏ qua label 0 = background)
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n_labels)]
    areas.sort(reverse=True)

    clean = np.zeros_like(binary, dtype=np.uint8)
    for _, idx in areas[:2]:
        component = (labels == idx).astype(np.uint8)
        # Convex hull mỗi lá phổi → solid đặc, không jagged edge, không lỗ nội bộ.
        # Cần thiết cho AP/portable X-ray trẻ em mà PSPNet cho mask răng cưa.
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(contours[0])
            cv2.fillConvexPoly(clean, hull, 1)
        else:
            clean[labels == idx] = 1

    # morphClose nhỏ để smooth cạnh convex hull
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k)
    return clean


def _hard_resize(binary: np.ndarray, out_size: int) -> np.ndarray:
    """INTER_NEAREST resize — exactly matches training get_lung_mask."""
    return cv2.resize(binary.astype(np.uint8), (out_size, out_size),
                      interpolation=cv2.INTER_NEAREST).astype(bool)



def get_image_transform():
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(settings.NORM_MEAN, settings.NORM_STD)
    ])


def get_lung_mask(img_pil: Image.Image, out_size: int = 448) -> np.ndarray:
    """
    Trả về hard binary bool mask (out_size × out_size) khớp training pipeline.
    Primary  : PSPNet (torchxrayvision) — threshold 0.5, dilate 15px, INTER_NEAREST
    Fallback : CV2 Otsu + morphology
    Fallback2: Thoracic ellipse cố định
    """
    # --- Primary: PSPNet ---
    try:
        import torchxrayvision as xrv
        from app.dependencies.model_loader import model_loader

        seg    = model_loader.seg_model
        ll_idx = model_loader.ll_idx
        rl_idx = model_loader.rl_idx

        if seg is not None and ll_idx is not None and rl_idx is not None:
            device = next(seg.parameters()).device
            g = np.array(img_pil.convert("L").resize((512, 512))).astype(np.float32)
            g = xrv.datasets.normalize(g, 255)
            t = torch.from_numpy(g)[None, None, ...].to(device)
            with torch.no_grad():
                out = torch.sigmoid(seg(t))

            lung = np.zeros((512, 512), dtype=np.uint8)
            for ch in [ll_idx, rl_idx]:
                prob = out[0, ch].cpu().numpy().astype(np.float32)
                prob = cv2.GaussianBlur(prob, (41, 41), 0)
                binary = (prob > 0.20).astype(np.uint8)
                if binary.sum() < 200:
                    continue

                # Lấy component lớn nhất → convexHull chỉ trên vùng đó
                # (tránh stray blobs xa làm hull quá rộng như khi dùng tất cả pixel)
                n_lab, labs, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                if n_lab <= 1:
                    continue
                largest_idx = max(range(1, n_lab), key=lambda i: stats[i, cv2.CC_STAT_AREA])
                component = (labs == largest_idx).astype(np.uint8)

                ys, xs = np.where(component > 0)
                if len(ys) < 3:
                    continue
                pts = np.column_stack([xs, ys]).reshape(-1, 1, 2).astype(np.int32)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(lung, hull, 1)

            if lung.sum() < 1000:
                raise ValueError(f"PSPNet detected too little lung area ({lung.sum()} px), using fallback")

            lung = cv2.dilate(lung,
                              np.ones((settings.DILATE_PX, settings.DILATE_PX), np.uint8),
                              iterations=1)
            return _hard_resize(lung, out_size)
    except Exception as e:
        logger.warning(f"PSPNet masking failed: {e}. Falling back to CV2.")

    # --- Fallback: CV2 ---
    try:
        gray = np.array(img_pil.convert("L"))
        h, w = gray.shape
        image_area = h * w

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl    = clahe.apply(gray)
        blur  = cv2.GaussianBlur(cl, (5, 5), 0)

        _, body = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
        k_body  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        body    = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k_body)

        _, dark      = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        lung_cand    = cv2.bitwise_and(dark, body)

        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(lung_cand, cv2.MORPH_OPEN,  k, iterations=2)
        cleaned = cv2.morphologyEx(cleaned,   cv2.MORPH_CLOSE, k, iterations=4)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binary_mask = np.zeros_like(gray, dtype=np.uint8)

        if not contours:
            raise ValueError("No contours")
        valid = sorted([c for c in contours if cv2.contourArea(c) > 0.01 * image_area],
                       key=cv2.contourArea, reverse=True)[:2]
        if not valid:
            raise ValueError("No valid contours")
        cv2.drawContours(binary_mask, valid, -1, 1, thickness=cv2.FILLED)
        binary_mask = _keep_two_lungs(binary_mask)   # fill holes

        binary_mask = cv2.dilate(binary_mask,
                                 np.ones((settings.DILATE_PX, settings.DILATE_PX), np.uint8),
                                 iterations=1)
        return _hard_resize(binary_mask, out_size)

    except Exception as e:
        logger.warning(f"CV2 fallback failed ({e}). Using thoracic ellipse.")
        bm = np.zeros((img_pil.height, img_pil.width), dtype=np.uint8)
        h2, w2 = bm.shape
        cv2.ellipse(bm, (int(w2*.28), int(h2*.45)), (int(w2*.18), int(h2*.32)), 0, 0, 360, 1, -1)
        cv2.ellipse(bm, (int(w2*.72), int(h2*.45)), (int(w2*.18), int(h2*.32)), 0, 0, 360, 1, -1)
        bm = cv2.dilate(bm,
                        np.ones((settings.DILATE_PX, settings.DILATE_PX), np.uint8),
                        iterations=1)
        return _hard_resize(bm, out_size)


def preprocess_image(image_bytes: bytes) -> tuple:
    """
    Returns:
        input_tensor     – tensor masked (hard mask, khớp training)  shape (1,3,448,448)
        img_display      – PIL 448×448 với soft-edge mask (chỉ dùng cho GradCAM overlay)
        input_tensor_tta – tensor horizontal-flip cho TTA
        lung_mask        – hard bool mask (448,448) cho lung_focus_ratio
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    target = settings.CENTER_CROP  # 448

    # 1. Lung mask — hard binary, khớp training
    lung_mask = get_lung_mask(image, out_size=target)

    # 2. Resize ảnh về 448×448
    img_arr = np.array(image.resize((target, target)))

    # Hard-masked PIL — dùng cho cả model input VÀ GradCAM overlay (khớp Kaggle pipeline)
    masked_arr = (img_arr * np.expand_dims(lung_mask, -1)).astype(np.uint8)
    masked_pil = Image.fromarray(masked_arr)

    normalize = transforms.Normalize(settings.NORM_MEAN, settings.NORM_STD)

    input_tensor = normalize(transforms.ToTensor()(masked_pil)).unsqueeze(0)
    input_tensor_tta = normalize(
        transforms.ToTensor()(masked_pil.transpose(Image.FLIP_LEFT_RIGHT))
    ).unsqueeze(0)

    # masked_pil làm nền GradCAM: đen ngoài phổi, lung tissue hiển thị — giống Kaggle
    return input_tensor, masked_pil, input_tensor_tta, lung_mask
