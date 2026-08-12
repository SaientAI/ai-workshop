#!/usr/bin/env bash
# Launch Saient.
# If the window is already open, focus it. Otherwise kill any stale port and start fresh.

APP_DIR="$(cd "$(dirname "$0")/llm-inference" && pwd)"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_TITLE="Saient"
PID_FILE="${XDG_RUNTIME_DIR:-/tmp}/saient-dev.pid"

focus_existing_window() {
    if command -v xdotool >/dev/null 2>&1; then
        local window_id
        window_id="$(xdotool search --name "^${APP_TITLE}$" 2>/dev/null | head -n 1)"
        [ -n "$window_id" ] || return 1
        xdotool windowfocus "$window_id" >/dev/null 2>&1 && return 0
        return 1
    fi

    if command -v wmctrl >/dev/null 2>&1; then
        local window_id
        window_id="$(wmctrl -l 2>/dev/null | awk -v title="$APP_TITLE" '$0 ~ (" " title "$") { print $1; exit }')"
        [ -n "$window_id" ] || return 1
        wmctrl -i -a "$window_id" >/dev/null 2>&1 && return 0
    fi

    return 1
}

is_running_pid() {
    [ -r "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

is_dev_app_running() {
    is_running_pid && return 0

    local pid cwd
    for pid in $(pgrep -f "target/debug/llm-inference" 2>/dev/null || true); do
        cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
        if [ "$cwd" = "$APP_DIR" ] || [ "$cwd" = "$APP_DIR/src-tauri" ]; then
            return 0
        fi
    done
    for pid in $(pgrep -f "tauri dev" 2>/dev/null || true); do
        cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
        if [ "$cwd" = "$APP_DIR" ] || [ "$cwd" = "$APP_DIR/src-tauri" ]; then
            return 0
        fi
    done

    return 1
}

# If the app is already running, just focus it
if focus_existing_window; then
    exit 0
fi

if is_dev_app_running; then
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "Saient" "Already running. Install wmctrl or xdotool to focus it from this launcher."
    fi
    exit 0
fi

# Kill any stale Vite dev server on port 1421. Only do this after proving the app
# itself is not running, otherwise repeat launcher clicks can break the session.
fuser -k 1421/tcp 2>/dev/null || true

cd "$APP_DIR"
export SAIENT_DATA_DIR="${SAIENT_DATA_DIR:-$ROOT_DIR/data}"
export SAIENT_CONFIG_DIR="${SAIENT_CONFIG_DIR:-$SAIENT_DATA_DIR/config/saient-dev}"
export SAIENT_MODELS_DIR="${SAIENT_MODELS_DIR:-$SAIENT_DATA_DIR/models}"
export SAIENT_RUNTIME_ASSETS_DIR="${SAIENT_RUNTIME_ASSETS_DIR:-$SAIENT_DATA_DIR/runtime-assets}"
export HF_HOME="${HF_HOME:-$SAIENT_DATA_DIR/runtime-tmp/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_XET=1
echo "$$" > "$PID_FILE"
exec npm run tauri dev
