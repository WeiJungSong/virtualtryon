"""Paths and tunables. Override any value with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
TOOLS_DIR = ROOT / ".tools"

# Keep every downloaded model inside the project folder (portable, and deleting the
# folder removes everything). Set these variables yourself to share a cache instead.
os.environ.setdefault("HF_HOME", str(TOOLS_DIR / "huggingface"))   # CatVTON weights
os.environ.setdefault("U2NET_HOME", str(TOOLS_DIR / "u2net"))      # rembg models
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DATA_DIR = Path(os.environ.get("VTON_DATA_DIR", ROOT / "data"))
GARMENT_DIR = DATA_DIR / "garments"
RESULT_DIR = DATA_DIR / "results"
TRASH_DIR = DATA_DIR / "trash"

HOST = os.environ.get("VTON_HOST", "127.0.0.1")
PORT = int(os.environ.get("VTON_PORT", "8765"))

# Background removal model used by rembg for product photos.
REMBG_MODEL = os.environ.get("VTON_REMBG_MODEL", "isnet-general-use")

# CatVTON weights (downloaded from Hugging Face on first use, ~2.3 GB total).
BASE_REPO = os.environ.get("VTON_BASE_REPO", "stable-diffusion-v1-5/stable-diffusion-inpainting")
VAE_REPO = os.environ.get("VTON_VAE_REPO", "stabilityai/sd-vae-ft-mse")
ATTN_REPO = os.environ.get("VTON_ATTN_REPO", "zhengchong/CatVTON")
ATTN_SUBFOLDER = os.environ.get("VTON_ATTN_SUBFOLDER", "mix-48k-1024/attention")

# Force a torch device ("mps", "cuda", "cpu"); empty = auto.
FORCE_DEVICE = os.environ.get("VTON_DEVICE", "")

for d in (GARMENT_DIR, RESULT_DIR, TRASH_DIR):
    d.mkdir(parents=True, exist_ok=True)
