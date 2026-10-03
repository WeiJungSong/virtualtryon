"""
CatVTON inference engine, adapted for Apple Silicon (MPS), CUDA and CPU.

The diffusion loop and the attention set-up follow the official implementation
(https://github.com/Zheng-Chong/CatVTON, CC BY-NC-SA 4.0, non-commercial use only):
SD-1.5 inpainting UNet, cross-attention removed, person and garment latents
concatenated along the height axis. Changes here: progress callbacks, chunked
self-attention to bound memory on MPS, a DPM-Solver option for fewer steps,
fp16 UNet with fp32 VAE, and compositing back into the full camera frame.
"""
from __future__ import annotations

import gc
import inspect
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from . import config

PRESETS = {
    # name: width, height (multiples of 64), steps, guidance, scheduler
    "fast": dict(width=384, height=512, steps=20, guidance=2.5, scheduler="dpm"),
    "standard": dict(width=576, height=768, steps=30, guidance=2.5, scheduler="dpm"),
    "high": dict(width=768, height=1024, steps=50, guidance=2.5, scheduler="ddim"),
}

# Files actually needed (keeps the download at ~2.3 GB instead of ~10 GB)
DOWNLOADS = [
    (config.BASE_REPO, ["unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                        "scheduler/scheduler_config.json"]),
    (config.VAE_REPO, ["config.json", "diffusion_pytorch_model.safetensors"]),
    (config.ATTN_REPO, [f"{config.ATTN_SUBFOLDER}/*"]),
]
EXPECTED_BYTES = 1_719_154_104 + 334_643_276 + 198_303_368


# --------------------------------------------------------------------------- #
# Attention processors (nn.Module subclasses on purpose: the CatVTON checkpoint
# is indexed by the order of modules whose name contains "attn1", and the
# official code registers its processors as modules, so we must do the same)
# --------------------------------------------------------------------------- #
def _make_processors():
    import torch
    import torch.nn.functional as F

    class SkipAttnProcessor(torch.nn.Module):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
            return hidden_states

    class ChunkedSelfAttnProcessor(torch.nn.Module):
        """SDPA self-attention, splitting queries so the score matrix stays ~<=1 GB."""

        budget_bytes = 1_000_000_000
        min_chunk = 256

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                b, c, h, w = hidden_states.shape
                hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
            batch, seq, _ = hidden_states.shape
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            q = attn.to_q(hidden_states)
            k = attn.to_k(context)
            v = attn.to_v(context)
            head_dim = k.shape[-1] // attn.heads
            q = q.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
            k = k.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
            v = v.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
            kv_len = k.shape[2]
            elem = q.element_size()
            chunk = max(self.min_chunk, int(self.budget_bytes // max(1, batch * attn.heads * kv_len * elem)))
            if seq <= chunk:
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                out = torch.cat(
                    [F.scaled_dot_product_attention(q[:, :, i:i + chunk], k, v) for i in range(0, seq, chunk)],
                    dim=2,
                )
            out = out.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(q.dtype)
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            if input_ndim == 4:
                out = out.transpose(-1, -2).reshape(b, c, h, w)
            if attn.residual_connection:
                out = out + residual
            return out / attn.rescale_output_factor

    return SkipAttnProcessor, ChunkedSelfAttnProcessor


def init_adapter(unet):
    Skip, SelfAttn = _make_processors()
    procs = {}
    for name in unet.attn_processors.keys():
        procs[name] = SelfAttn() if name.endswith("attn1.processor") else Skip()
    unet.set_attn_processor(procs)


def attention_modules(unet):
    import torch
    blocks = torch.nn.ModuleList()
    for name, module in unet.named_modules():
        if "attn1" in name:
            blocks.append(module)
    return blocks


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def resize_and_padding(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = image.size
    tw, th = size
    if w / h < tw / th:
        nh, nw = th, max(1, w * th // h)
    else:
        nw, nh = tw, max(1, h * tw // w)
    image = image.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(image, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def crop_padded(img: Image.Image, box: tuple[int, int, int, int], fill_edge: bool) -> Image.Image:
    """Crop a box that may extend past the image borders (edge-replicate or black)."""
    x0, y0, x1, y1 = box
    arr = np.array(img)
    h, w = arr.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    if pl or pt or pr or pb:
        pad = [(pt, pb), (pl, pr)] + ([(0, 0)] if arr.ndim == 3 else [])
        arr = np.pad(arr, pad, mode="edge" if fill_edge else "constant")
    arr = arr[y0 + pt:y1 + pt, x0 + pl:x1 + pl]
    return Image.fromarray(arr)


def default_crop_box(mask: Image.Image) -> tuple[int, int, int, int]:
    m = np.array(mask.convert("L")) > 127
    w, h = mask.size
    ys, xs = np.where(m)
    if len(xs) == 0:
        cx, cy, bh = w / 2, h / 2, h
    else:
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        bh = max((ys.max() - ys.min()) * 1.3, (xs.max() - xs.min()) * 4 / 3 * 1.15)
    bh = max(bh, 256)
    bw = bh * 3 / 4
    return int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)


def smooth_mask(mask: Image.Image) -> Image.Image:
    """Blur + binarise (same effect as CatVTON's VaeImageProcessor.blur(mask, 9))."""
    m = mask.convert("L").filter(ImageFilter.GaussianBlur(9))
    return m.point(lambda p: 255 if p >= 128 else 0)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class CatVTONEngine:
    def __init__(self):
        self.state = "not_loaded"  # not_loaded | downloading | loading | ready | error
        self.message = ""
        self.error: str | None = None
        self.device = None
        self.dtype = None
        self.unet = self.vae = None
        self._sched_cfg_path = None
        self._lock = threading.Lock()
        self.download_bytes = 0
        self.load_seconds = None

    # ---- environment --------------------------------------------------------
    @staticmethod
    def pick_device():
        import torch
        forced = config.FORCE_DEVICE
        if forced:
            dev = forced
        elif torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        dtype = torch.float32 if dev == "cpu" else torch.float16
        return dev, dtype

    def info(self) -> dict:
        try:
            dev, _ = self.pick_device()
        except Exception as e:  # torch missing
            dev = f"unavailable ({e})"
        return {
            "state": self.state,
            "message": self.message,
            "error": self.error,
            "device": self.device or dev,
            "download_mb": round(self.download_bytes / 1e6),
            "expected_mb": round(EXPECTED_BYTES / 1e6),
            "presets": PRESETS,
        }

    # ---- download -----------------------------------------------------------
    def _download(self) -> dict[str, Path]:
        from huggingface_hub import snapshot_download, try_to_load_from_cache  # noqa: F401
        from huggingface_hub.constants import HF_HUB_CACHE

        paths: dict[str, Path] = {}
        stop = threading.Event()

        def watch():
            cache = Path(HF_HUB_CACHE)
            dirs = [cache / ("models--" + repo.replace("/", "--")) for repo, _ in DOWNLOADS]
            while not stop.wait(1.0):
                total = 0
                for d in dirs:
                    if d.exists():
                        total += sum(f.stat().st_size for f in (d / "blobs").glob("*") if f.is_file()) if (d / "blobs").exists() else 0
                self.download_bytes = total

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        try:
            for repo, patterns in DOWNLOADS:
                self.message = f"下載模型 {repo} …"
                paths[repo] = Path(snapshot_download(repo_id=repo, allow_patterns=patterns))
        finally:
            stop.set()
        return paths

    # ---- load ---------------------------------------------------------------
    def load(self):
        with self._lock:
            if self.state == "ready":
                return
            t0 = time.time()
            try:
                import torch
                from accelerate import load_checkpoint_in_model
                from diffusers import AutoencoderKL, UNet2DConditionModel

                self.device, self.dtype = self.pick_device()
                self.state, self.error = "downloading", None
                paths = self._download()
                self.state = "loading"
                self.message = f"載入模型到 {self.device} …"
                base = paths[config.BASE_REPO]
                unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", variant="fp16", torch_dtype=torch.float32)
                init_adapter(unet)
                attn = attention_modules(unet)
                load_checkpoint_in_model(attn, str(paths[config.ATTN_REPO] / config.ATTN_SUBFOLDER))
                self.unet = unet.to(self.device, dtype=self.dtype).eval()
                vae = AutoencoderKL.from_pretrained(paths[config.VAE_REPO], torch_dtype=torch.float32)
                self.vae = vae.to(self.device).eval()
                self._sched_dir = base
                self.state = "ready"
                self.load_seconds = round(time.time() - t0, 1)
                self.message = f"模型就緒（{self.device}，{self.load_seconds}s）"
            except Exception as e:
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
                self.message = "模型載入失敗"
                raise

    def load_from_modules(self, unet, vae, scheduler_dir, device="cpu", dtype=None):
        """Testing hook: use already constructed modules (same dtype policy as load())."""
        import torch
        init_adapter(unet)
        self.device, self.dtype = device, dtype or torch.float32
        self.unet = unet.to(device, dtype=self.dtype).eval()
        self.vae = vae.to(device, dtype=torch.float32).eval()
        self._sched_dir = Path(scheduler_dir)
        self.state = "ready"

    def _scheduler(self, kind: str):
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
        cls = DPMSolverMultistepScheduler if kind == "dpm" else DDIMScheduler
        return cls.from_pretrained(self._sched_dir, subfolder="scheduler")

    # ---- inference ----------------------------------------------------------
    def _encode(self, img_tensor):
        import torch
        with torch.no_grad():
            lat = self.vae.encode(img_tensor.to(self.device, torch.float32)).latent_dist.sample()
        return lat * self.vae.config.scaling_factor

    def _to_tensor(self, img: Image.Image):
        import torch
        arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)[None]

    def generate(
        self,
        person: Image.Image,
        garment: Image.Image,
        mask: Image.Image,
        preset: str = "fast",
        seed: int = 42,
        progress: Callable[[int, int], None] | None = None,
    ) -> Image.Image:
        """person/mask are already cropped to 3:4; returns an image at model resolution."""
        import torch
        from diffusers.utils.torch_utils import randn_tensor

        if self.state != "ready":
            self.load()
        p = PRESETS.get(preset, PRESETS["fast"])
        W, H = p["width"], p["height"]
        person = person.convert("RGB").resize((W, H), Image.LANCZOS)
        mask = smooth_mask(mask.resize((W, H), Image.NEAREST))
        garment = resize_and_padding(garment.convert("RGB"), (W, H))

        image_t = self._to_tensor(person)
        cond_t = self._to_tensor(garment)
        mask_t = torch.from_numpy((np.array(mask) >= 128).astype(np.float32))[None, None]
        masked = image_t * (mask_t < 0.5)

        def run(unet_dtype):
            with torch.no_grad():
                masked_lat = self._encode(masked).to(unet_dtype)
                cond_lat = self._encode(cond_t).to(unet_dtype)
                mask_lat = torch.nn.functional.interpolate(mask_t, size=masked_lat.shape[-2:], mode="nearest").to(self.device, unet_dtype)
                cat = -2  # concatenate along height
                masked_cat = torch.cat([masked_lat, cond_lat], dim=cat)
                mask_cat = torch.cat([mask_lat, torch.zeros_like(mask_lat)], dim=cat)
                gen = torch.Generator(device="cpu").manual_seed(int(seed))
                latents = randn_tensor(masked_cat.shape, generator=gen, device=torch.device(self.device), dtype=unet_dtype)
                sched = self._scheduler(p["scheduler"])
                sched.set_timesteps(p["steps"], device=self.device)
                latents = latents * sched.init_noise_sigma
                cfg = p["guidance"] > 1.0
                if cfg:
                    masked_cat = torch.cat([torch.cat([masked_lat, torch.zeros_like(cond_lat)], dim=cat), masked_cat])
                    mask_cat = torch.cat([mask_cat] * 2)
                extra = {}
                params = set(inspect.signature(sched.step).parameters)
                if "eta" in params:
                    extra["eta"] = 1.0
                if "generator" in params:
                    extra["generator"] = gen
                timesteps = sched.timesteps
                for i, t in enumerate(timesteps):
                    x = torch.cat([latents] * 2) if cfg else latents
                    x = sched.scale_model_input(x, t)
                    x = torch.cat([x, mask_cat, masked_cat], dim=1)
                    noise = self.unet(x, t.to(self.device), encoder_hidden_states=None, return_dict=False)[0]
                    if cfg:
                        nu, nc = noise.chunk(2)
                        noise = nu + p["guidance"] * (nc - nu)
                    latents = sched.step(noise, t, latents, **extra).prev_sample
                    if progress:
                        progress(i + 1, len(timesteps))
                latents = latents.split(latents.shape[cat] // 2, dim=cat)[0]
                latents = latents.to(torch.float32) / self.vae.config.scaling_factor
                out = self.vae.decode(latents).sample
                return (out / 2 + 0.5).clamp(0, 1)

        out = run(self.dtype)
        if torch.isnan(out).any() and self.dtype != torch.float32:
            # fp16 overflow on some MPS setups: fall back to fp32 permanently
            self.unet = self.unet.to(torch.float32)
            self.dtype = torch.float32
            out = run(self.dtype)
        arr = (out[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
        self._cleanup()
        return Image.fromarray(arr)

    def _cleanup(self):
        import torch
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    # ---- full-frame helper --------------------------------------------------
    def tryon_frame(
        self,
        frame: Image.Image,
        mask: Image.Image,
        garment: Image.Image,
        crop_box: tuple[int, int, int, int] | None,
        preset: str,
        seed: int,
        progress=None,
    ) -> tuple[Image.Image, Image.Image]:
        """Run on a camera frame and paste the result back. Returns (composite, crop_result)."""
        frame = frame.convert("RGB")
        mask = mask.convert("L").resize(frame.size, Image.NEAREST)
        box = crop_box or default_crop_box(mask)
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        # enforce 3:4
        if abs(bw / bh - 0.75) > 0.01:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bh = max(bh, bw * 4 / 3)
            bw = bh * 3 / 4
            x0, y0, x1, y1 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
        person_c = crop_padded(frame, (x0, y0, x1, y1), fill_edge=True)
        mask_c = crop_padded(mask, (x0, y0, x1, y1), fill_edge=False)
        result = self.generate(person_c, garment, mask_c, preset=preset, seed=seed, progress=progress)

        res_full = result.resize(person_c.size, Image.LANCZOS)
        k = max(3, int(min(person_c.size) * 0.012) | 1)
        feather = smooth_mask(mask_c).filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.GaussianBlur(k / 2))
        blended = Image.composite(res_full, person_c, feather)
        out = frame.copy()
        # paste only the part that lies inside the frame
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(frame.width, x1), min(frame.height, y1)
        if fx1 > fx0 and fy1 > fy0:
            region = blended.crop((fx0 - x0, fy0 - y0, fx1 - x0, fy1 - y0))
            out.paste(region, (fx0, fy0))
        return out, result
