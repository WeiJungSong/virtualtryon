#!/usr/bin/env bash
# Local Try-On installer for macOS (Apple Silicon / Intel) and Linux.
#   bash setup.sh          install
#   bash setup.sh --hd     install and also download + test the HD model (~2.3 GB)
# Everything goes inside this folder (.tools/ and .venv/); nothing is installed system-wide.
#
# Written for bash 3.2 (the macOS default): every variable is braced, because bash 3.2
# treats UTF-8 bytes right after $NAME as part of the variable name.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
TOOLS="${ROOT}/.tools"
PYVER="3.11"

say() { printf '%s\n' "$*"; }
die() { printf '\n[錯誤] %s\n' "$*" >&2; exit 1; }

say "=== 本機虛擬試穿 安裝 ==="

# ---------------------------------------------------------------------------
# Platform. On Apple Silicon everything must run as arm64 (PyTorch has no new
# Intel-Mac builds); sysctl tells the truth even when this Terminal uses Rosetta.
# ---------------------------------------------------------------------------
OS="$(uname -s)"
APPLE_SILICON=0
if [ "${OS}" = "Darwin" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
  APPLE_SILICON=1
fi
native() {
  if [ "${APPLE_SILICON}" = "1" ]; then arch -arm64 "$@"; else "$@"; fi
}
case "${OS}" in
  Darwin)
    if [ "${APPLE_SILICON}" = "1" ]; then TARGET="aarch64-apple-darwin"; else TARGET="x86_64-apple-darwin"; fi ;;
  Linux)
    case "$(uname -m)" in
      aarch64|arm64) TARGET="aarch64-unknown-linux-gnu" ;;
      x86_64|amd64) TARGET="x86_64-unknown-linux-gnu" ;;
      *) die "不支援的 CPU 架構" ;;
    esac ;;
  *) die "不支援的作業系統（Windows 請執行 setup.bat）" ;;
esac
say "   系統：${OS} · ${TARGET}"

command -v curl >/dev/null 2>&1 || die "找不到 curl"
command -v tar >/dev/null 2>&1 || die "找不到 tar"

# uv wheel on PyPI for this platform (PyPI's CDN is usually much faster than GitHub)
case "${TARGET}" in
  aarch64-apple-darwin) WHL_PAT='macosx_[0-9_]*_arm64\.whl' ;;
  x86_64-apple-darwin) WHL_PAT='macosx_[0-9_]*_x86_64\.whl' ;;
  aarch64-unknown-linux-gnu) WHL_PAT='manylinux_2_17_aarch64[^"]*\.whl' ;;
  x86_64-unknown-linux-gnu) WHL_PAT='manylinux_2_17_x86_64[^"]*\.whl' ;;
esac

# (no head / grep -m below: they exit early and trip pipefail with SIGPIPE)
uv_from_pypi() {
  command -v unzip >/dev/null 2>&1 || return 1
  local ver url
  ver="$(curl -fsSL --connect-timeout 20 https://pypi.org/pypi/uv/json | grep -o '"version":"[^"]*"' | sed -n '1p' | cut -d'"' -f4)" || return 1
  [ -n "${ver}" ] || return 1
  url="$(curl -fsSL --connect-timeout 20 "https://pypi.org/pypi/uv/${ver}/json" | tr ',' '\n' | grep -o '"url":"[^"]*'"${WHL_PAT}"'"' | sed -n '1p' | cut -d'"' -f4)" || return 1
  [ -n "${url}" ] || return 1
  say "   來源：PyPI（uv ${ver}）"
  curl -fL --progress-bar --retry 3 --connect-timeout 20 -o "${TMPD}/uv.whl" "${url}" || return 1
  unzip -o -q "${TMPD}/uv.whl" "uv-${ver}.data/scripts/uv" -d "${TMPD}/whl" || return 1
  cp "${TMPD}/whl/uv-${ver}.data/scripts/uv" "${UV}"
}

uv_from_github() {
  local url="https://github.com/astral-sh/uv/releases/latest/download/uv-${TARGET}.tar.gz"
  say "   來源：GitHub"
  curl -fL --progress-bar --retry 3 --connect-timeout 20 -o "${TMPD}/uv.tar.gz" "${url}" || return 1
  tar -xzf "${TMPD}/uv.tar.gz" -C "${TMPD}" || return 1
  cp "${TMPD}/uv-${TARGET}/uv" "${UV}"
}

export UV_PYTHON_INSTALL_DIR="${TOOLS}/python"
export UV_PYTHON_BIN_DIR="${TOOLS}/python-bin"   # keep uv's python3.x links out of ~/.local/bin
export UV_PYTHON_PREFERENCE="only-managed"
export UV_CACHE_DIR="${TOOLS}/cache"
export UV_NO_MODIFY_PATH=1
export INSTALLER_NO_MODIFY_PATH=1

# ---------------------------------------------------------------------------
# 1) uv (downloads Python and packages; a single file in .tools/uv)
# ---------------------------------------------------------------------------
UV="${TOOLS}/uv/uv"
if [ -n "${VTON_UV:-}" ]; then
  UV="${VTON_UV}"   # use an existing uv (offline / managed machines)
  say "① 使用指定的 uv：${UV}"
elif [ ! -x "${UV}" ]; then
  say "① 下載 uv（套件管理工具）"
  mkdir -p "${TOOLS}/uv"
  TMPD="$(mktemp -d)"
  if uv_from_pypi; then :
  elif uv_from_github; then :
  else
    say "   改用官方安裝程式"
    native env UV_INSTALL_DIR="${TOOLS}/uv" sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' || die "無法下載 uv，請確認網路連線"
  fi
  chmod +x "${UV}"
else
  say "① uv 已就緒"
fi
native "${UV}" --version

# ---------------------------------------------------------------------------
# 2) Python (a private copy, native for this CPU)
# ---------------------------------------------------------------------------
if [ -n "${VTON_PYTHON:-}" ]; then
  PYSEL="${VTON_PYTHON}"   # use an existing Python 3.10-3.13 instead of downloading one
  export UV_PYTHON_PREFERENCE="system"
  MARK="custom-${TARGET}"
  say "② 使用指定的 Python：${PYSEL}"
else
  PYSEL="${PYVER}"
  MARK="${PYVER}-${TARGET}"
  say "② 準備 Python ${PYVER}"
  native "${UV}" python install "${PYVER}"
fi

if [ -d .venv ] && [ "$(cat .venv/.vton-python 2>/dev/null || true)" != "${MARK}" ]; then
  OLD=".venv.old-$(date +%Y%m%d-%H%M%S)"
  mv .venv "${OLD}"
  say "   舊的 .venv 不相容，已改名為 ${OLD}（確認不需要後可自行刪除）"
fi
if [ ! -x .venv/bin/python ]; then
  native "${UV}" venv --python "${PYSEL}" .venv
fi
printf '%s\n' "${MARK}" > .venv/.vton-python

# ---------------------------------------------------------------------------
# 3+) PyTorch, packages, assets, self test (Python does the platform logic)
# ---------------------------------------------------------------------------
export UV_BIN="${UV}"
set +e
native .venv/bin/python tools/install.py ${1+"$@"}
RC=$?
set -e
chmod +x ./run.sh ./setup.sh ./*.command 2>/dev/null || true
say ""
if [ "${RC}" -eq 0 ]; then
  say "安裝完成。啟動：bash run.sh（或雙擊 run.command）"
else
  say "安裝沒有完全成功（代碼 ${RC}），請把上面的訊息複製給我。"
fi
exit "${RC}"
