import os
import numpy as np
import torch
import joblib
from PIL import Image, ImageFilter
from transformers import CLIPModel, CLIPProcessor

# xet downloads have been observed to leave a dangling/stub safetensors file
# for this large CLIP checkpoint -> disable xet so the legacy downloader is used.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

CLIP_ID = "openai/clip-vit-large-patch14"
_CACHE = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
_HERE = os.path.dirname(os.path.abspath(__file__))

CLF_REPO = "Strawbercar/AIDetector_ClipCLF"
CLF_FILENAME = "clip_clf.joblib"
CLF_PATH = os.path.join(_HERE, CLF_FILENAME)

FAKE_CLASS = 1
FAKE_THRESHOLD = 0.75  # prob_fake >= this -> "artificial", else "real"
_FEAT_LAYER = -2  

_model = None
_processor = None
_clf = None


def _load():
    global _model, _processor
    if _model is None:
        _processor = CLIPProcessor.from_pretrained(CLIP_ID, cache_dir=_CACHE)
        _model = CLIPModel.from_pretrained(CLIP_ID, cache_dir=_CACHE)
        _model.eval()
    return _model, _processor


def _to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise TypeError("expected a path or PIL.Image")


def _load_clf():
    global _clf
    if _clf is None:
        if os.path.exists(CLF_PATH):
            path = CLF_PATH
        elif CLF_REPO:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=CLF_REPO, filename=CLF_FILENAME,
                                   cache_dir=_CACHE)
        else:
            raise RuntimeError(
                "No trained CLIP classifier found. Place "
                f"'{CLF_FILENAME}' next to ClipDetectInfer.py or set CLF_REPO.")
        _clf = joblib.load(path)
    return _clf


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


def detect_ai_clip(image):
    clf = _load_clf()
    clip_model, processor = _load()
    img = _to_pil(image)

    inputs = processor(images=img, return_tensors="pt")
    pv = inputs["pixel_values"]
    pv.requires_grad_(True)

    with torch.enable_grad():
        hs = clip_model.vision_model(pixel_values=pv, output_hidden_states=True).hidden_states
        feat = hs[_FEAT_LAYER][0, 0, :]            # penultimate CLS, with grad
        feat_n = (feat / feat.norm()).detach().cpu().numpy().reshape(1, -1)

    prob_fake = float(clf.predict_proba(feat_n)[0, FAKE_CLASS])

    # sklearn binary LR stores coef_/intercept_ for class index 1. The fake
    # logit is +decision_function if fake is class 1, else its negation.
    coef = torch.from_numpy(clf.coef_[0])
    intercept = float(clf.intercept_[0])
    logit1 = (coef * feat).sum() + intercept
    fake_logit = logit1 if FAKE_CLASS == 1 else -logit1

    prediction = "artificial" if prob_fake >= FAKE_THRESHOLD else "real"

    clip_model.zero_grad()
    fake_logit.backward()
    sal = pv.grad[0].abs().mean(0).detach().cpu().numpy()
    mx = sal.max()
    sal = sal / mx if mx > 0 else sal

    h, w = sal.shape
    sal_u8 = (sal * 255).astype(np.uint8)
    sal_u8 = np.array(Image.fromarray(sal_u8).filter(ImageFilter.GaussianBlur(2)))
    sal = sal_u8.astype(np.float32) / 255.0
    mx = sal.max()
    if mx > 0:
        sal = sal / mx

    heatmap = _colorize(sal)
    base = img.resize((w, h), Image.BILINEAR).convert("RGB")
    overlay = Image.blend(base, heatmap, 0.5)

    return overlay, prediction, prob_fake