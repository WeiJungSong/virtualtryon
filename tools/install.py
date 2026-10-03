"""
Installer step 2 (runs inside .venv, standard library only).
Called by setup.sh / setup.bat after uv + Python are ready.

  python tools/install.py [--hd] [--skip-selftest]

Picks the right PyTorch build for this machine, installs requirements,
downloads runtime assets and runs the self test.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
UV = os.environ.get("UV_BIN") or shutil.which("uv")


def run(cmd: list[str]) -> int:
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd])


def uv_pip(*args: str) -> int:
    return run([UV, "pip", "install", "--python", PY, *args])


def has_nvidia() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def torch_plan() -> tuple[list[str], str | None, Path | None, str]:
    """(packages, index_url, constraints_file, description)

    Override with VTON_TORCH_SPEC (e.g. "torch==2.6.0") and/or VTON_TORCH_INDEX
    (e.g. an AMD ROCm index) for unusual machines.
    """
    machine = platform.machine().lower()
    if os.environ.get("VTON_TORCH_SPEC") or os.environ.get("VTON_TORCH_INDEX"):
        spec = os.environ.get("VTON_TORCH_SPEC", "torch")
        return [spec], os.environ.get("VTON_TORCH_INDEX") or None, None, f"自訂：{spec}"
    if sys.platform == "darwin":
        if machine == "arm64":
            return ["torch"], None, None, "Apple Silicon：PyTorch（Apple GPU / MPS）"
        return ["torch==2.2.2"], None, ROOT / "constraints" / "intel-mac.txt", \
            "Intel Mac：PyTorch 2.2.2（Intel Mac 的最後一版）"
    if has_nvidia():
        return ["torch"], "https://download.pytorch.org/whl/cu126", None, "NVIDIA GPU：PyTorch CUDA 12.6"
    if sys.platform.startswith("linux"):
        # PyPI's Linux wheels bundle CUDA (several GB); the CPU index is much smaller
        return ["torch"], "https://download.pytorch.org/whl/cpu", None, "沒有 NVIDIA GPU：PyTorch CPU 版"
    return ["torch"], None, None, "沒有 NVIDIA GPU：PyTorch CPU 版"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", action="store_true", help="也下載並測試 HD 模型（約 2.3 GB）")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()
    if not UV:
        print("[錯誤] 找不到 uv（請用 setup.sh / setup.bat 執行安裝）")
        return 1

    print(f"   Python {platform.python_version()} · {sys.platform} · {platform.machine()}")
    pkgs, index, constraints, desc = torch_plan()
    print(f"③ 安裝 PyTorch — {desc}")
    cflags = ["-c", str(constraints)] if constraints else []
    rc = 1
    if index:
        rc = uv_pip(*pkgs, "--index-url", index, *cflags)
        if rc != 0:
            print("   官方 PyTorch 套件庫連不上，改從 PyPI 安裝")
    if rc != 0:
        rc = uv_pip(*pkgs, *cflags)
    if rc != 0:
        print("[錯誤] PyTorch 安裝失敗")
        return 1

    print("④ 安裝其他套件")
    if uv_pip("-r", str(ROOT / "requirements.txt"), *cflags) != 0:
        print("[錯誤] 套件安裝失敗")
        return 1

    print("⑤ 下載前端模型與去背模型")
    fetch = [PY, str(ROOT / "tools" / "fetch_assets.py")] + (["--hd"] if args.hd else [])
    if run(fetch) != 0:
        print("[警告] 部分資產下載失敗，網頁會改從 CDN 載入")

    if args.skip_selftest:
        return 0
    print("⑥ 自我檢測")
    test = [PY, str(ROOT / "tools" / "selftest.py")] + (["--hd"] if args.hd else [])
    return run(test)


if __name__ == "__main__":
    sys.exit(main())
