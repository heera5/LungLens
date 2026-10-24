"""LungLens - CT scan analysis for the Flask app.
Loads ct_model.pt (made by the Colab notebook), predicts Normal / Benign / Malignant
and draws a Grad-CAM heatmap. torch is imported lazily, so the rest of the app
works even if PyTorch or the model file is missing."""
import io, os, base64, threading, importlib.util
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ct_model.pt')
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224
LOW_CONFIDENCE = 0.60
MAX_COLOR_SPREAD = 8.0          # real CT slices are grey; photos/screenshots are not

_lock = threading.Lock()
_state = {'model': None, 'arch': None, 'classes': None, 'metrics': None, 'core': None, 'torch': None}


class ModelUnavailable(Exception):
    pass


class InvalidImage(Exception):
    pass


def torch_installed():
    return importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None


def status():
    """Return (ready, message) for the upload page."""
    if not os.path.exists(MODEL_PATH):
        return False, 'The CT model file (ct_model.pt) was not found. Train it with the Colab notebook and copy it into the FRONTEND folder.'
    if not torch_installed():
        return False, 'PyTorch is not installed. Install torch and torchvision to enable CT analysis.'
    return True, ''


def _ensure_loaded():
    ready, msg = status()
    if not ready:
        raise ModelUnavailable(msg)
    with _lock:
        if _state['model'] is not None:
            return
        import torch
        import ct_core
        ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        model = ct_core.build_model(ckpt['arch'], num_classes=len(ckpt['classes']), pretrained=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        _state.update(model=model, arch=ckpt['arch'], classes=list(ckpt['classes']),
                      metrics=ckpt.get('metrics', {}), core=ct_core, torch=torch)


def model_info():
    if _state['model'] is None:
        try:
            _ensure_loaded()
        except ModelUnavailable:
            return None
    return {'arch': _state['arch'], 'classes': _state['classes'], 'metrics': _state['metrics']}


# ---------------- image helpers (no torch needed, easy to test) ----------------
def open_image(data):
    """Validate and open uploaded bytes as an RGB PIL image."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        raise InvalidImage('The file is not a valid image.')
    if min(img.size) < 64:
        raise InvalidImage('The image is too small (minimum 64 x 64 pixels).')
    return img


def color_spread(img):
    a = np.asarray(img.resize((128, 128)), dtype=np.float32)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())


def to_array(img):
    """PIL RGB -> normalised float array [1,3,H,W] (same preprocessing as in training)."""
    a = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def _b64_jpeg(img, quality=88):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def make_overlay(img, cam, alpha=0.45, max_side=512):
    """Blend a Grad-CAM map (values 0..1) onto the image. Returns (original_b64, overlay_b64)."""
    import cv2
    disp = img.copy()
    disp.thumbnail((max_side, max_side))
    w, h = disp.size
    cam_u8 = (np.clip(cv2.resize(cam.astype(np.float32), (w, h)), 0, 1) * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    blend = ((1 - alpha) * np.asarray(disp, dtype=np.float32) + alpha * heat).astype(np.uint8)
    return _b64_jpeg(disp), _b64_jpeg(Image.fromarray(blend))


# ---------------- prediction ----------------
def predict(data):
    img = open_image(data)
    _ensure_loaded()
    torch, core = _state['torch'], _state['core']
    x = torch.from_numpy(to_array(img))
    cam, probs, idx = core.grad_cam(_state['model'], _state['arch'], x)
    classes = _state['classes']
    original_b64, overlay_b64 = make_overlay(img, cam)
    conf = float(probs[idx])
    return {
        'label': classes[idx],
        'confidence': conf,
        'probs': [{'label': c, 'prob': float(p)} for c, p in zip(classes, probs)],
        'low_confidence': conf < LOW_CONFIDENCE,
        'not_grayscale': color_spread(img) > MAX_COLOR_SPREAD,
        'original_b64': original_b64,
        'overlay_b64': overlay_b64,
        'arch': _state['arch'],
    }
