#!/bin/bash
# macOS: double-click to download the HD model (~2.3 GB, once) and test it on this Mac.
cd "$(dirname "$0")/.." || exit 1
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
  arch -arm64 .venv/bin/python tools/selftest.py --hd
else
  .venv/bin/python tools/selftest.py --hd
fi
echo
read -n 1 -s -r -p "按任意鍵關閉視窗…"
echo
