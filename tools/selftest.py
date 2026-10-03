"""
Post-install self test. Exit code 0 means the app is ready to use.

  python tools/selftest.py          # environment, PyTorch device, HD engine (tiny test model),
                                    # background removal, web assets, server API
  python tools/selftest.py --hd     # also run the real CatVTON weights once (~2.3 GB download)
"""
from __future__ import annotations

import argparse
import importlib
import io
import os
import platform
import socket
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
TMP = Path(tempfile.mkdtemp(prefix="vton-selftest-"))
os.environ["VTON_DATA_DIR"] = str(TMP / "data")  # never touch the real wardrobe

results: list[tuple[str, str, str]] = []  # (status ok|warn|fail, name, detail)


def check(name: str, warn_only: bool = False):
    def deco(fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            status = "ok"
            if isinstance(detail, tuple):
                status, detail = detail
        except Exception as e:  # noqa: BLE001
            status = "warn" if warn_only else "fail"
            detail = f"{type(e).__name__}: {e}"
            if os.environ.get("VTON_SELFTEST_TRACE"):
                traceback.print_exc()
        mark = {"ok": "✓", "warn": "!", "fail": "✗"}[status]
        print(f"   {mark} {name}  {detail}  ({time.time() - t0:.1f}s)", flush=True)
        results.append((status, name, str(detail)))
        return fn
    return deco


def shirt_png() -> bytes:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (600, 700), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.polygon([(150, 80), (250, 50), (350, 50), (450, 80), (580, 230), (500, 300), (450, 250),
               (450, 650), (150, 650), (150, 250), (100, 300), (20, 230)], fill=(200, 60, 70))
    d.ellipse((250, 30, 350, 110), fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", action="store_true")
    args = ap.parse_args()
    print(f"   測試資料暫存於 {TMP}")

    @check("Python")
    def _():
        if sys.version_info < (3, 10):
            raise RuntimeError(f"需要 3.10 以上，目前 {platform.python_version()}")
        mach = platform.machine()
        if sys.platform == "darwin":
            import subprocess
            arm = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True).stdout.strip()
            if arm == "1" and mach != "arm64":
                raise RuntimeError(f"這台是 Apple Silicon，但 Python 以 {mach}（Rosetta）執行")
        return f"{platform.python_version()} · {sys.platform} · {mach}"

    @check("套件")
    def _():
        mods = ["fastapi", "uvicorn", "multipart", "httpx", "bs4", "PIL", "numpy", "cv2",
                "onnxruntime", "torch", "diffusers", "accelerate", "huggingface_hub", "safetensors"]
        missing = []
        for m in mods:
            try:
                importlib.import_module(m)
            except Exception as e:  # noqa: BLE001
                missing.append(f"{m}（{type(e).__name__}: {e}）")
        if missing:
            raise RuntimeError("無法匯入：" + "；".join(missing))
        return f"{len(mods)} 個套件"

    from server.catvton_engine import CatVTONEngine
    try:
        from diffusers.utils import logging as dlog
        dlog.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass

    dev_info = {}

    @check("PyTorch 裝置")
    def _():
        import torch
        dev, dtype = CatVTONEngine.pick_device()
        x = torch.randn(512, 512, device=dev, dtype=dtype)
        y = (x @ x).float().abs().mean().item()
        if not y == y:  # NaN
            raise RuntimeError("矩陣運算得到 NaN")
        dev_info.update(dev=dev, dtype=dtype)
        note = {"mps": "Apple GPU", "cuda": "NVIDIA GPU", "cpu": "CPU（HD 生成會很慢）"}.get(dev, dev)
        status = "warn" if dev == "cpu" else "ok"
        return status, f"PyTorch {torch.__version__} · {dev} · {str(dtype).replace('torch.', '')} · {note}"

    @check("HD 引擎（小型測試模型，不需下載）")
    def _():
        import numpy as np
        from PIL import Image, ImageDraw
        from tiny_models import scheduler_dir, tiny_unet, tiny_vae
        from server import catvton_engine as ce
        eng = CatVTONEngine()
        eng.load_from_modules(tiny_unet(), tiny_vae(), scheduler_dir(TMP), device=dev_info.get("dev", "cpu"),
                              dtype=dev_info.get("dtype"))
        ce.PRESETS["selftest"] = dict(width=128, height=192, steps=3, guidance=2.5, scheduler="dpm")
        frame = Image.new("RGB", (640, 360), (90, 120, 160))
        mask = Image.new("L", (640, 360), 0)
        ImageDraw.Draw(mask).rectangle([260, 80, 380, 300], fill=255)
        garment = Image.open(io.BytesIO(shirt_png())).convert("RGB")
        comp, crop = eng.tryon_frame(frame, mask, garment, (220, 20, 420, 287), "selftest", 1)
        a = np.asarray(comp)
        if comp.size != frame.size or not np.isfinite(a).all():
            raise RuntimeError("輸出異常")
        fell_back = "（fp16 溢位已自動改用 fp32）" if str(eng.dtype) != str(dev_info.get("dtype")) else ""
        return f"{eng.device} · {str(eng.dtype).replace('torch.', '')} {fell_back}"

    @check("去背模型（ISNet）", warn_only=True)
    def _():
        from server import garment_proc as gp
        path = gp.rembg_model_path()
        if path is not None and not path.exists():
            return "warn", "尚未下載完成，程式啟動後會在背景續傳（期間白底商品圖用內建去背）"
        if gp._get_rembg(block=True, timeout=300) is None:
            raise RuntimeError(f"去背模型無法載入：{gp.rembg_status()}（白底商品圖仍可用內建去背）")
        return "isnet-general-use"

    @check("前端資產", warn_only=True)
    def _():
        import json
        man = ROOT / "web" / "vendor" / "manifest.json"
        if not man.exists():
            return "warn", "未下載，網頁會從 CDN 載入（需連網）"
        m = json.loads(man.read_text())
        if m.get("tasks_vision") and m.get("pose") and m.get("seg"):
            return "本機（可離線使用）"
        return "warn", f"部分缺少 {m}，缺少的部分會從 CDN 載入"

    @check("伺服器與 API")
    def _():
        import httpx
        import uvicorn
        from server import app as appmod
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        srv = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=port, log_level="error"))
        th = threading.Thread(target=srv.run, daemon=True)
        th.start()
        for _ in range(100):
            if srv.started:
                break
            time.sleep(0.1)
        base = f"http://127.0.0.1:{port}"
        H = {"X-Requested-With": "vton"}
        try:
            with httpx.Client(timeout=120) as c:
                assert c.get(base + "/api/status", headers=H).status_code == 200, "/api/status"
                assert c.get(base + "/api/status").status_code == 403, "缺少防護標頭時應拒絕"
                page = c.get(base + "/")
                assert page.status_code == 200 and "本機虛擬試穿" in page.text, "首頁"
                js = c.get(base + "/js/main.js")
                assert "javascript" in js.headers.get("content-type", ""), f"JS MIME：{js.headers.get('content-type')}"
                assert js.headers.get("cache-control") == "no-cache", "程式檔應每次重新驗證（更新後免強制重新整理）"
                vend = ROOT / "web" / "vendor" / "tasks-vision"
                if (vend / "vision_bundle.mjs").exists():
                    ct = c.get(base + "/vendor/tasks-vision/vision_bundle.mjs").headers.get("content-type", "")
                    assert "javascript" in ct, f".mjs MIME：{ct}"
                    ct = c.head(base + "/vendor/tasks-vision/wasm/vision_wasm_internal.wasm").headers.get("content-type", "")
                    assert "wasm" in ct, f".wasm MIME：{ct}"
                r = c.post(base + "/api/garments", headers=H, data={"category": "upper"},
                           files={"file": ("shirt.png", shirt_png(), "image/png")})
                assert r.status_code == 200, f"上傳衣服：{r.status_code} {r.text[:200]}"
                matting = r.json().get("matting")
                cut = c.get(base + r.json()["cutout_url"])
                assert cut.status_code == 200, "去背圖"
                lst = c.get(base + "/api/garments", headers=H).json()["garments"]
                assert len(lst) == 1, "衣櫃清單"
        finally:
            srv.should_exit = True
            th.join(10)
        return f"首頁、API、上傳去背（{matting}）皆正常"

    if args.hd:
        @check("HD 實際模型（CatVTON，快速）")
        def _():
            import numpy as np
            from PIL import Image, ImageDraw
            eng = CatVTONEngine()
            t0 = time.time()
            eng.load()
            load_s = time.time() - t0
            person = Image.new("RGB", (384, 512), (150, 140, 130))
            d = ImageDraw.Draw(person)
            d.ellipse((150, 30, 234, 120), fill=(220, 180, 160))
            d.rectangle((110, 130, 274, 400), fill=(240, 240, 240))
            mask = Image.new("L", (384, 512), 0)
            ImageDraw.Draw(mask).rectangle((90, 120, 294, 420), fill=255)
            garment = Image.open(io.BytesIO(shirt_png())).convert("RGB")
            t1 = time.time()
            out = eng.generate(person, garment, mask, preset="fast", seed=1)
            gen_s = time.time() - t1
            a = np.asarray(out).astype(float)
            if a.std() < 2:
                raise RuntimeError("輸出是單色圖")
            out.save(TMP / "hd_selftest.jpg")
            return f"{eng.device} · 載入 {load_s:.0f}s · 生成 {gen_s:.0f}s · 結果 {TMP / 'hd_selftest.jpg'}"

    fails = [r for r in results if r[0] == "fail"]
    warns = [r for r in results if r[0] == "warn"]
    print()
    if fails:
        print(f"   自我檢測：{len(fails)} 項失敗 —— " + "；".join(f"{n}：{d}" for _, n, d in fails))
        return 1
    print("   自我檢測：全部通過" + (f"（{len(warns)} 項提醒）" if warns else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
