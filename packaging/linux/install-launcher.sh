#!/usr/bin/env bash
# Install a Saient desktop launcher (app menu + clickable Desktop icon) that points
# at the locally-built release binary. Idempotent; safe to re-run after a rebuild.
#
#   ./packaging/linux/install-launcher.sh
#
set -euo pipefail

# Repo root = two levels up from this script.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

bin="$repo/llm-inference/src-tauri/target/release/llm-inference"
icon_src="$repo/llm-inference/src-tauri/icons/128x128.png"
tmpl="$here/Saient.desktop"

if [ ! -x "$bin" ]; then
  echo "✗ release binary not found at:" >&2
  echo "    $bin" >&2
  echo "  Build it first:  (cd llm-inference && npm run tauri build -- --bundles deb,appimage)" >&2
  exit 1
fi

apps="$HOME/.local/share/applications"
icons="$HOME/.local/share/icons/hicolor/128x128/apps"
mkdir -p "$apps" "$icons"

# Icon
cp "$icon_src" "$icons/saient.png"

# Render the launcher from the template with the real binary path.
launcher="$apps/Saient.desktop"
sed "s#__SAIENT_BIN__#$bin#" "$tmpl" > "$launcher"

# Retire the stale pre-rebrand launcher if present (same binary, old "AI Workshop" name).
rm -f "$apps/AI Workshop.desktop"

update-desktop-database "$apps" 2>/dev/null || true
gtk-update-icon-cache -f "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

# Clickable icon on the Desktop (marked trusted so file managers launch it directly).
if [ -d "$HOME/Desktop" ]; then
  cp "$launcher" "$HOME/Desktop/Saient.desktop"
  chmod +x "$HOME/Desktop/Saient.desktop"
  gio set "$HOME/Desktop/Saient.desktop" metadata::trusted true 2>/dev/null || true
fi

echo "✓ Saient launcher installed → $launcher"
echo "  Desktop icon: $HOME/Desktop/Saient.desktop"
echo "  Points at:    $bin"
