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


def analyze_spectrum(image, size=256, alpha=0.0):
    """Return (spectrum_image, summary_text)."""
    img = _to_pil(image).convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)

    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a))))
    mx = mag.max()
    norm = mag / (mx + 1e-8)
    spec = _colorize(norm).resize((size, size), Image.NEAREST)

    h, w = norm.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = min(h, w) * 0.3
    hf_ratio = float(norm[r > cutoff].sum()) / float(norm.sum() + 1e-8)

    note = ("Very low high-frequency energy... image may be over-smoothed (AI / denoised)."
            if hf_ratio < 0.05 else "High-frequency energy is high — look for grid-like peaks "
            "(a sign of GAN upsampling)." if hf_ratio > 0.25 else
            "High-frequency energy is in a normal range.")

    summary = (f"High-frequency energy ratio: {hf_ratio:.3f}  (cutoff radius = 30% of frame)\n"
               f"Read: {note}\n"
               "Center = low frequency (large structure), edges = high frequency (fine detail/edges). "
               "GAN images can leave sharp spectral peaks; modern diffusion models usually don't, "
               "so this is a weak signal on its own - best used alongside the other tools.")
    return spec, summary