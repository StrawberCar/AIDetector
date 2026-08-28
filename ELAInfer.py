import io
import numpy as np
from PIL import Image


def _to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise TypeError("expected a path or PIL.Image")


def _jet(v):
    v = min(1.0, max(0.0, v))
    if v < 0.25:
        return 0, int(4 * v * 255), 255
    if v < 0.5:
        return 0, 255, int((1 - 4 * (v - 0.25)) * 255)
    if v < 0.75:
        return int(4 * (v - 0.5) * 255), 255, 0
    return 255, int((1 - 4 * (v - 0.75)) * 255), 0


def _colorize(cam):
    h, w = cam.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            rgb[y, x] = _jet(float(cam[y, x]))
    return Image.fromarray(rgb, "RGB")


def analyze_ela(image, quality=85, alpha=0.55):
    """Return (heatmap_overlay, summary_text)."""
    img = _to_pil(image)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(resaved, dtype=np.float32)
    diff = np.abs(a - b).mean(axis=2)            # grayscale per-pixel error
    mx = diff.max()
    norm = diff / (mx + 1e-8)

    heat = _colorize(norm).resize(img.size, Image.BILINEAR)
    overlay = Image.blend(img.convert("RGB"), heat, float(alpha))

    mean_err = float(diff.mean())
    std_err = float(diff.std())
    # high mean with low spread = uniformly high error = never cleanly JPEG'd
    uniform = mean_err / (std_err + 1e-8)
    flag = ("UNIFORM high error across the whole image.. consistent with an image that was "
            "never a clean camera JPEG (AI export, PNG, or recompressed)."
            if uniform > 2.2 and mean_err > 6 else
            "Error is patchy / concentrated... look for localized bright regions (edited/pasted areas)."
            if std_err > 4 else
            "Low, smooth error... image compresses like a normal JPEG.")

    summary = (f"JPEG re-save quality: {int(quality)}\n"
              f"Mean error: {mean_err:.2f}/255   Std: {std_err:.2f}   Uniformity: {uniform:.2f}\n"
              f"Read: {flag}")
    return overlay, summary