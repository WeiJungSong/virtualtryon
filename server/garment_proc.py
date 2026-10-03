"""Garment image acquisition and background removal."""
from __future__ import annotations

import io
import ipaddress
import os
import json
import re
import socket
import threading
from pathlib import Path
from urllib.parse import urljoin, urlparse

import cv2
import httpx
import numpy as np
from PIL import Image, ImageOps

from . import config

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
MAX_DOWNLOAD = 25 * 1024 * 1024


class FetchError(Exception):
    pass


# --------------------------------------------------------------------------- #
# URL fetching (with a guard against requests into the local network)
# --------------------------------------------------------------------------- #
def check_public_url(url: str) -> str:
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https") or not p.hostname:
        raise FetchError("只支援 http / https 網址")
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror as e:
        raise FetchError(f"無法解析網域：{p.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise FetchError("不允許存取區域網路位址")
    return url.strip()


def fetch(url: str, referer: str | None = None) -> tuple[bytes, str, str]:
    """Return (content, content_type, final_url)."""
    url = check_public_url(url)
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,image/*,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    try:
        with httpx.Client(follow_redirects=True, timeout=20, headers=headers) as client:
            with client.stream("GET", url) as r:
                check_public_url(str(r.url))
                if r.status_code >= 400:
                    raise FetchError(f"網站回應 {r.status_code}")
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > MAX_DOWNLOAD:
                        raise FetchError("檔案太大")
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                return bytes(buf), ctype, str(r.url)
    except httpx.HTTPError as e:
        raise FetchError(f"下載失敗：{e}") from e


def looks_like_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


_IMG_EXT = re.compile(r"\.(jpe?g|png|webp|avif)(\?|$)", re.I)
_SKIP = re.compile(r"(sprite|icon|logo|favicon|placeholder|blank|pixel|tracking|badge|avatar|\.svg|\.gif)", re.I)


def _best_from_srcset(srcset: str) -> str | None:
    best, best_w = None, -1
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        w = 0
        if len(bits) > 1:
            m = re.match(r"(\d+)(w|x)", bits[1])
            if m:
                w = int(m.group(1)) * (1000 if m.group(2) == "x" else 1)
        if w >= best_w:
            best, best_w = bits[0], w
    return best


def extract_page_images(html: str, base_url: str, limit: int = 40) -> tuple[str, list[str]]:
    """Find likely product images in a product page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    found: list[str] = []

    def add(u: str | None):
        if not u:
            return
        u = u.strip()
        if u.startswith("data:"):
            return
        u = urljoin(base_url, u)
        if _SKIP.search(u):
            return
        if u not in found:
            found.append(u)

    for prop in ("og:image", "og:image:secure_url", "twitter:image", "twitter:image:src"):
        for tag in soup.find_all("meta", attrs={"property": prop}) + soup.find_all("meta", attrs={"name": prop}):
            add(tag.get("content"))
    for tag in soup.find_all("link", rel=lambda v: v and "image_src" in v):
        add(tag.get("href"))
    # JSON-LD Product.image
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                img = node.get("image")
                if isinstance(img, str):
                    add(img)
                elif isinstance(img, list):
                    for i in img:
                        add(i if isinstance(i, str) else (i or {}).get("url"))
                elif isinstance(img, dict):
                    add(img.get("url"))
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
    for img in soup.find_all(["img", "source"]):
        for attr in ("data-zoom-image", "data-large", "data-src", "data-original", "src"):
            v = img.get(attr)
            if v and (_IMG_EXT.search(v) or attr != "src"):
                add(v)
                break
        for attr in ("srcset", "data-srcset"):
            if img.get(attr):
                add(_best_from_srcset(img.get(attr)))
    return title, found[:limit]


# --------------------------------------------------------------------------- #
# Background removal
# --------------------------------------------------------------------------- #
_rembg_lock = threading.Lock()
_rembg_session = None
_rembg_error: str | None = None
_rembg_state = "idle"          # idle | downloading | loading | ready | error
_rembg_progress = (0, 0)       # bytes done, total

# Known model files: (file name, sha256, mirrors). The Hugging Face copy is byte-identical
# to rembg's GitHub release (same sha256) and usually downloads much faster.
REMBG_FILES = {
    "isnet-general-use": (
        "isnet-general-use.onnx",
        "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        [
            "https://huggingface.co/tomjackson2023/rembg/resolve/main/isnet-general-use.onnx",
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        ],
    ),
}


def rembg_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


class IsNetMatting:
    """
    ISNet background removal run directly with onnxruntime. Same preprocessing and
    mask post-processing as rembg's "isnet-general-use" session (rembg remove(...,
    post_process_mask=True)), without rembg's heavy dependencies (numba, scipy, ...).
    """

    def __init__(self, model_path: Path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        self.sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def alpha(self, img: Image.Image) -> np.ndarray:
        im = img.convert("RGB").resize((1024, 1024), Image.LANCZOS)
        a = np.asarray(im).astype(np.float64)
        a = a / max(a.max(), 1e-6)
        x = ((a - 0.5) / 1.0).transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.sess.run(None, {self.inp: x})[0][:, 0, :, :]
        pred = np.squeeze((pred - pred.min()) / max(pred.max() - pred.min(), 1e-12))
        mask = Image.fromarray((pred * 255).astype(np.uint8), "L").resize(img.size, Image.LANCZOS)
        m = np.asarray(mask)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        m = cv2.GaussianBlur(m.astype(np.float32), (17, 17), 2, borderType=cv2.BORDER_REFLECT)
        return np.where(m < 127, 0, 255).astype(np.uint8)


def rembg_model_path() -> Path | None:
    spec = REMBG_FILES.get(config.REMBG_MODEL)
    if not spec:
        return None
    # rembg looks for <U2NET_HOME>/<name>.onnx (old flat layout) in every version
    return Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net")) / spec[0]


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_rembg_model(deadline: float | None = None, verbose: bool = False) -> bool:
    """Download the rembg model with resume + sha256 check. False if not finished by `deadline`."""
    global _rembg_progress
    import time
    import urllib.request
    target = rembg_model_path()
    if target is None:
        return True  # unknown model: let rembg download it itself
    if target.exists():
        return True
    fname, sha, urls = REMBG_FILES[config.REMBG_MODEL]
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".onnx.part")
    progress_line = False   # a "\r ↓ ..." line is open on the terminal
    for url in urls:
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "virtual-tryon", **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=30) as r:
                if have and r.status != 206:      # server ignored Range: start over
                    have = 0
                total = have + int(r.headers.get("content-length") or 0)
                with open(part, "ab" if have else "wb") as f:
                    last = 0.0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        _rembg_progress = (have, total)
                        if verbose and time.time() - last > 1:
                            last = time.time()
                            print(f"\r  ↓ {fname} {have / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
                            progress_line = True
                        if deadline and time.time() > deadline:
                            if verbose and progress_line:
                                print()
                            return False
            if verbose and progress_line:
                print(f"\r  ↓ {fname} {have / 1e6:.0f}/{total / 1e6:.0f} MB")
                progress_line = False
            if _sha256(part) == sha:
                part.replace(target)
                return True
            part.unlink()  # corrupt / different file: try the next mirror from scratch
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"{chr(10) if progress_line else ''}  ✗ {url.split('/')[2]}：{e}")
                progress_line = False
    return target.exists()


def _load_rembg():
    global _rembg_session, _rembg_error, _rembg_state
    try:
        _rembg_state = "downloading"
        ensure_rembg_model()
        _rembg_state = "loading"
        path = rembg_model_path()
        if path is not None:
            sess = IsNetMatting(path)
        else:  # other rembg models (VTON_REMBG_MODEL): needs the optional rembg package
            from rembg import new_session
            sess = new_session(config.REMBG_MODEL)
        with _rembg_lock:
            _rembg_session = sess
        _rembg_state = "ready"
    except Exception as e:  # noqa: BLE001
        _rembg_error = f"{type(e).__name__}: {e}"
        _rembg_state = "error"


def start_rembg_warmup():
    """Load (and if needed download) the model in the background; never blocks."""
    global _rembg_state
    with _rembg_lock:
        if _rembg_state != "idle":
            return
        _rembg_state = "downloading"
    threading.Thread(target=_load_rembg, daemon=True).start()


def rembg_status() -> dict:
    done, total = _rembg_progress
    return {"state": _rembg_state, "error": _rembg_error,
            "progress_mb": round(done / 1e6), "total_mb": round(total / 1e6)}


def _get_rembg(block: bool = False, timeout: float = 600):
    """Return the session if ready. Non-blocking by default: starts the warm-up and
    returns None so callers fall back to the built-in matting meanwhile."""
    import time
    start_rembg_warmup()
    if block:
        t0 = time.time()
        while _rembg_state in ("downloading", "loading") and time.time() - t0 < timeout:
            time.sleep(0.2)
    return _rembg_session


def _studio_alpha(rgb: np.ndarray, strict: bool = True) -> np.ndarray | None:
    """
    Matting for product shots on a plain background (the usual e-commerce photo).
    Background = pixels close to the border colour AND connected to the border.
    Generic segmentation models often drop light garments on white backgrounds;
    this catches them because even a cream knit differs measurably from #FFFFFF.
    Returns None when the background is not plain.
    """
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bw = max(2, min(h, w) // 200)
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
                             lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    bdist = np.linalg.norm(border - bg, axis=1)
    noise = float(np.percentile(bdist, 95))
    if strict and (noise > 8 or np.mean(bdist < max(6, noise * 1.5)) < 0.85):
        return None  # busy / non-uniform background
    dist = np.linalg.norm(lab - bg, axis=2)
    t = max(5.0, noise * 2.0 + 3.0)
    cand = (dist <= t).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels = cv2.connectedComponents(cand, connectivity=4)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge_labels = edge_labels[edge_labels != 0]
    bgmask = np.isin(labels, edge_labels)
    alpha = np.where(bgmask, 0, 255).astype(np.uint8)
    # soft edge from the colour distance inside a thin band around the boundary
    band = cv2.dilate(alpha, np.ones((3, 3), np.uint8)) != cv2.erode(alpha, np.ones((3, 3), np.uint8))
    soft = np.clip((dist - t * 0.5) / (t * 1.5), 0, 1) * 255
    alpha = np.where(band, np.maximum(alpha * 0.5, soft), alpha).astype(np.uint8)
    return alpha


def _clean_alpha(alpha: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    binary = (alpha > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_min = max(areas.max() * 0.04, h * w * 0.0005)
        keep = np.zeros(n, bool)
        keep[1:] = areas >= keep_min
        alpha = np.where(keep[labels], alpha, 0).astype(np.uint8)
    # trim light fringe left by white backgrounds
    alpha = cv2.erode(alpha, np.ones((2, 2), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return alpha


def remove_background(img: Image.Image) -> tuple[Image.Image, str]:
    """Return (RGBA cut-out cropped to the garment, method)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1536, 1536), Image.LANCZOS)
    rgb = np.array(img)
    methods = []
    model_alpha = None
    session = _get_rembg()
    if session is not None:
        try:
            if isinstance(session, IsNetMatting):
                model_alpha = session.alpha(img)
            else:
                from rembg import remove
                model_alpha = np.array(remove(img, session=session, post_process_mask=True).convert("RGBA"))[..., 3]
            methods.append(config.REMBG_MODEL)
        except Exception:
            model_alpha = None
    studio = _studio_alpha(rgb)
    if studio is not None:
        methods.append("studio")
        alpha = studio if model_alpha is None else np.maximum(studio, model_alpha)
    elif model_alpha is not None and model_alpha.max() >= 128:
        alpha = model_alpha
    else:
        alpha = _studio_alpha(rgb, strict=False)
        methods.append("floodfill")
    method = "+".join(methods)
    alpha = _clean_alpha(alpha)
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0:
        raise ValueError("找不到衣服主體，請換一張背景較單純的商品圖")
    pad = int(0.03 * max(rgb.shape[:2]))
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
    rgba = np.dstack([rgb, alpha])[y0:y1, x0:x1]
    return Image.fromarray(rgba, "RGBA"), method


def flatten_on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg
