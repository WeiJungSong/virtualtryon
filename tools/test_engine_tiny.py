"""
Smoke-test the CatVTON engine with tiny random weights (no downloads needed).

Parts 1-2 compare against the official implementation; they need a clone of
https://github.com/Zheng-Chong/CatVTON  (set CATVTON_REPO=/path/to/CatVTON).
"""
import json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from server.catvton_engine import CatVTONEngine, init_adapter, attention_modules
from server import catvton_engine as ce

def tiny_unet():
    return UNet2DConditionModel(
        sample_size=32, in_channels=9, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, attention_head_dim=4,
    )

def tiny_vae():
    return AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                         up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8),
                         latent_channels=4, layers_per_block=1, norm_num_groups=4)

tmp = Path(tempfile.mkdtemp())
REPO = os.environ.get("CATVTON_REPO", "")
if not REPO or not Path(REPO, "model", "pipeline.py").exists():
    print("CATVTON_REPO not set: skipping the comparison with the official code")
    REPO = None
if REPO:
  # 1) checkpoint compatibility: build the attention ModuleList exactly like the official repo and save it
  sys.path.insert(0, REPO)
  from model.utils import init_adapter as off_init, get_trainable_module
  from model.attn_processor import SkipAttnProcessor as OffSkip
  from accelerate import Accelerator, load_checkpoint_in_model
  torch.manual_seed(0)
  ref = tiny_unet()
  off_init(ref, cross_attn_cls=OffSkip)
  ref_attn = get_trainable_module(ref, "attention")
  for p in ref_attn.parameters():
      p.data.normal_()
  ckdir = tmp / "attention"
  Accelerator().save_model(ref_attn, ckdir)
  print("saved keys:", len(json.load(open(ckdir / "model.safetensors.index.json"))["weight_map"]) if (ckdir / "model.safetensors.index.json").exists() else "single file")

  mine = tiny_unet()
  init_adapter(mine)
  load_checkpoint_in_model(attention_modules(mine), str(ckdir))
  sd_ref = {k: v for k, v in ref.state_dict().items() if "attn1" in k}
  sd_mine = {k: v for k, v in mine.state_dict().items() if "attn1" in k}
  assert sd_ref.keys() == sd_mine.keys()
  maxdiff = max((sd_ref[k] - sd_mine[k]).abs().max().item() for k in sd_ref)
  print("attn1 params:", len(sd_ref), "max diff after load:", maxdiff)
  assert maxdiff == 0.0

  # 2) forward-equivalence: official processors vs mine on the same input
  x = torch.randn(2, 9, 32, 24); t = torch.tensor(500)
  with torch.no_grad():
      a = ref(x, t, encoder_hidden_states=None, return_dict=False)[0]
      mine.load_state_dict(ref.state_dict())
      b = mine(x, t, encoder_hidden_states=None, return_dict=False)[0]
  print("forward max diff vs official:", (a - b).abs().max().item())
  assert (a - b).abs().max().item() < 1e-4

  # 3) chunked attention path == unchunked
  from server import catvton_engine as ce
  for name, proc in mine.attn_processors.items():
      if name.endswith("attn1.processor"):
          proc.budget_bytes = 10_000; proc.min_chunk = 16  # force chunking
  with torch.no_grad():
      c = mine(x, t, encoder_hidden_states=None, return_dict=False)[0]
  print("chunked vs unchunked max diff:", (b - c).abs().max().item())
  assert (b - c).abs().max().item() < 1e-4

# 4) full frame pipeline with tiny models
sched_dir = tmp / "base"
DDIMScheduler().save_pretrained(sched_dir / "scheduler")
eng = CatVTONEngine()
u = tiny_unet(); v = tiny_vae()
eng.load_from_modules(u, v, sched_dir)
ce.PRESETS["tiny"] = dict(width=128, height=192, steps=4, guidance=2.5, scheduler="ddim")
ce.PRESETS["tiny_dpm"] = dict(width=128, height=192, steps=4, guidance=2.5, scheduler="dpm")
frame = Image.new("RGB", (640, 360), (90, 120, 160))
mask = Image.new("L", (640, 360), 0); ImageDraw.Draw(mask).rectangle([260, 80, 380, 300], fill=255)
garment = Image.new("RGB", (300, 400), (200, 30, 30))
steps = []
for preset in ("tiny", "tiny_dpm"):
    t0 = time.time()
    comp, crop = eng.tryon_frame(frame, mask, garment, (220, 20, 420, 287), preset, 1, progress=lambda i, n: steps.append((i, n)))
    print(preset, "composite", comp.size, "crop", crop.size, "steps", steps[-1], f"{time.time()-t0:.2f}s")
    a = np.array(comp).astype(int); f = np.array(frame).astype(int)
    outside = np.abs(a[:, :200] - f[:, :200]).max()
    print("  unchanged outside crop:", outside == 0, "changed inside mask:", np.abs(a[150, 320] - f[150, 320]).sum() > 0)
# crop box beyond the frame edges (padding path)
comp, crop = eng.tryon_frame(frame, mask, garment, (-100, -60, 300, 473), "tiny", 3)
print("padded crop ok", comp.size)
comp, crop = eng.tryon_frame(frame, mask, garment, None, "tiny", 3)
print("auto crop ok", comp.size)
print("ALL OK")
