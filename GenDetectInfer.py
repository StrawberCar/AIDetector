import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# avoid xet leaving dangling/stub weight files on download
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_ID = "haywoodsloan/ai-image-detector-dev-deploy"
_CACHE = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")

_model = None
_processor = None

def _load():
    global _model, _processor
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=_CACHE)
        _model = AutoModelForImageClassification.from_pretrained(
            MODEL_ID, dtype=torch.float32, cache_dir=_CACHE
        )
        _model.eval()
    return _model, _processor


def _jet(v):
    """matplotlib-free 'jet' colormap, v in [0, 1] -> (r, g, b) ints."""
    v = min(1.0, max(0.0, v))
    if v < 0.25:
        return 0, int(4 * v * 255), 255
    if v < 0.5:
        return 0, 255, int((1 - 4 * (v - 0.25)) * 255)
    if v < 0.75:
        return int(4 * (v - 0.5) * 255), 255, 0
    return 255, int((1 - 4 * (v - 0.75)) * 255), 0


def _colorize(cam):
    """(H, W) float in [0, 1] -> jet-colored RGB PIL image."""
    h, w = cam.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            rgb[y, x] = _jet(float(cam[y, x]))
    return Image.fromarray(rgb, "RGB")


def _grad_cam(model, inputs, target_class, stage=4):
    out = model(**inputs, output_hidden_states=True)
    feats = out.hidden_states[stage]
    feats.retain_grad()
    b, n, c = feats.shape
    side = int(round(n ** 0.5))
    if side * side != n:
        raise RuntimeError(f"hidden state has {n} tokens, not a perfect square")

    model.zero_grad()
    out.logits[0, target_class].backward()
    grads = feats.grad
    feats = feats.view(b, side, side, c)
    grads = grads.view(b, side, side, c)

    weights = grads[0].mean(dim=(0, 1))
    cam = torch.relu((feats[0] * weights).sum(-1))
    cam = cam.detach().numpy().astype(np.float32)
    mx = cam.max()
    return cam / mx if mx > 0 else cam


def detect_ai(image, stage=4):
    model, processor = _load()

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError("`image` must be a path or a PIL.Image")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits[0], dim=0)
    pred_id = int(probs.argmax().item())
    prediction = model.config.id2label[pred_id]
    probability = float(probs[pred_id])

    label2id = {v: k for k, v in model.config.id2label.items()}
    target = label2id.get("artificial", pred_id)

    with torch.enable_grad():
        cam = _grad_cam(model, inputs, target, int(stage))

    w, h = inputs["pixel_values"].shape[-1], inputs["pixel_values"].shape[-2]
    size = (w, h)
    heatmap = _colorize(cam).resize(size, Image.BILINEAR)
    base = image.resize(size, Image.BILINEAR).convert("RGB")
    overlay = Image.blend(base, heatmap, 0.5)

    return overlay, prediction, probability