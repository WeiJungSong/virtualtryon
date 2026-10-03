"""
Download runtime assets so the app works offline afterwards.

  python tools/fetch_assets.py          # MediaPipe runtime + models, rembg model
  python tools/fetch_assets.py --hd     # also the CatVTON weights (~2.3 GB)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "web" / "vendor"
TV = "1.0.1"
CDN = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TV}"
FILES = {
    "tasks-vision/vision_bundle.mjs": f"{CDN}/vision_bundle.mjs",
    "tasks-vision/wasm/vision_wasm_internal.js": f"{CDN}/wasm/vision_wasm_internal.js",
    "tasks-vision/wasm/vision_wasm_internal.wasm": f"{CDN}/wasm/vision_wasm_internal.wasm",
    "tasks-vision/wasm/vision_wasm_module_internal.js": f"{CDN}/wasm/vision_wasm_module_internal.js",
    "tasks-vision/wasm/vision_wasm_module_internal.wasm": f"{CDN}/wasm/vision_wasm_module_internal.wasm",
    "tasks-vision/wasm/vision_wasm_nosimd_internal.js": f"{CDN}/wasm/vision_wasm_nosimd_internal.js",
    "tasks-vision/wasm/vision_wasm_nosimd_internal.wasm": f"{CDN}/wasm/vision_wasm_nosimd_internal.wasm",
    "models/pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "models/selfie_multiclass_256x256.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ 已存在 {dest.relative_to(ROOT)}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "virtual-tryon-setup"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("content-length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  ↓ {dest.name} {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
        tmp.replace(dest)
        print(f"\r  ✓ {dest.relative_to(ROOT)}{' ' * 20}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\r  ✗ {dest.name}: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", action="store_true", help="也下載 CatVTON 權重（約 2.3 GB）")
    args = ap.parse_args()

    print("① MediaPipe 執行環境與模型")
    ok = {rel: download(url, VENDOR / rel) for rel, url in FILES.items()}
    manifest = {
        "tasks_vision": TV if all(v for k, v in ok.items() if k.startswith("tasks-vision")) else None,
        "pose": ok["models/pose_landmarker_full.task"],
        "seg": ok["models/selfie_multiclass_256x256.tflite"],
    }
    (VENDOR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not all(ok.values()):
        print("  部分檔案下載失敗：網頁會改從 CDN 載入，功能不受影響（需連網）。")

    print("② 去背模型（ISNet）")
    try:
        import time
        sys.path.insert(0, str(ROOT))
        from server import config  # noqa: F401  (sets U2NET_HOME inside .tools/)
        from server import garment_proc as gp
        done = gp.ensure_rembg_model(deadline=time.time() + 300, verbose=True)
        if done:
            gp._get_rembg(block=True, timeout=120)
            st = gp.rembg_status()
            print(f"  ✓ {config.REMBG_MODEL}" if st["state"] == "ready" else f"  ✗ {st}")
        else:
            print("  … 網路較慢，剩下的部分會在程式啟動後於背景續傳（期間白底商品圖用內建去背）")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ rembg：{e}（啟動後會在背景重試）")

    if args.hd:
        print("③ CatVTON 權重（約 2.3 GB，下載到 Hugging Face 快取）")
        from huggingface_hub import snapshot_download
        from server.catvton_engine import DOWNLOADS
        for repo, patterns in DOWNLOADS:
            print(f"  ↓ {repo}")
            snapshot_download(repo_id=repo, allow_patterns=patterns)
        print("  ✓ 完成")
    print("資產準備完成。")


if __name__ == "__main__":
    main()
