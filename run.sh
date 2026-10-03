#!/usr/bin/env bash
# Start Local Try-On and open it in the browser (macOS / Linux).
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ ! -f .venv/.vton-python ]; then
  echo "尚未安裝，請先執行：bash setup.sh"
  exit 1
fi
APPLE_SILICON=0
if [ "$(uname -s)" = "Darwin" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
  APPLE_SILICON=1
fi
native() {
  if [ "${APPLE_SILICON}" = "1" ]; then arch -arm64 "$@"; else "$@"; fi
}

export PYTORCH_ENABLE_MPS_FALLBACK=1
PORT="${VTON_PORT:-8765}"
URL="http://127.0.0.1:${PORT}"

open_browser() {
  case "$(uname -s)" in
    Darwin)
      if [ -d "/Applications/Google Chrome.app" ]; then open -a "Google Chrome" "${URL}"; else open "${URL}"; fi ;;
    *)
      (xdg-open "${URL}" >/dev/null 2>&1 || true) ;;
  esac
}

if [ "${VTON_NO_BROWSER:-0}" != "1" ]; then
  (
    i=0
    while [ "${i}" -lt 60 ]; do
      sleep 0.5
      if curl -s -o /dev/null "${URL}"; then open_browser; break; fi
      i=$((i + 1))
    done
  ) &
fi

native .venv/bin/python -m server.app
