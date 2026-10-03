#!/bin/bash
# macOS: double-click to install (first time: right-click > Open if macOS blocks it).
cd "$(dirname "$0")" || exit 1
bash ./setup.sh
echo
read -n 1 -s -r -p "按任意鍵關閉視窗…"
echo
