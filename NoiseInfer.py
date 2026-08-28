import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


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


def analyze_noise(image, window=7, alpha=0.55):
    """Return (noise_variance_overlay, summary_text)."""
    img = _to_pil(image)
    arr = np.asarray(img, dtype=np.float32)

    denoised = np.asarray(img.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    residual = np.abs(arr - denoised).mean(axis=2)        # noise residual

    mean = ndimage.uniform_filter(residual, window)
    mean_sq = ndimage.uniform_filter(residual ** 2, window)
    var = np.clip(mean_sq - mean ** 2, 0, None)          # local noise variance

    mx = var.max()
    norm = var / (mx + 1e-8)
    heat = _colorize(norm).resize(img.size, Image.BILINEAR)
    overlay = Image.blend(img.convert("RGB"), heat, float(alpha))

    m = float(residual.mean())
    spread = float(var.std())
    clean = ("LOW... image looks suspiciously clean (possible AI / over-denoised)."
             if m < 3.0 else "normal-ish.")
    summary = (f"Mean residual: {m:.2f}/255   Local-var spread: {spread:.2f}\n"
               f"Residual level: {clean}\n"
               "Red = noisy patches, blue = smooth patches. Real camera photos have fairly "
               "uniform sensor noise; AI images often look too clean (low residual) or have "
               "inconsistent noise that breaks across regions.")
    return overlay, summary