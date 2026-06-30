#!/usr/bin/env bash
#
# Run the audioguide from start to finish :
#   1. start streaming (streaming_start.py)
#   2. run pipeline (main.py) that connects to the active flux
#
# Usage :
#   ./run.sh                             # WiFi, profile18 (default) — IP auto-detected via ARP
#   ./run.sh usb                         # USB, profile18
#   ./run.sh wifi profile18              # WiFi + explicit profile, IP auto-detected
#   ./run.sh wifi profile18 172.20.10.2  # WiFi + explicit IP, skips ARP scan
#
# Dependencies: adb, aria CLI, net-tools (arp for WiFi). Installed automatically where possible.
#
set -euo pipefail

cd "$(dirname "$0")"

OS="$(uname -s)"   # Linux or Darwin (macOS)
INTERFACE="${1:-wifi}"
PROFILE="${2:-profile18}"
DEVICE_IP="${3:-}"

# ping with a 1-second timeout, cross-platform
ping_once() {
    if [[ "$OS" == "Darwin" ]]; then
        ping -c1 -t1 "$1" &>/dev/null
    else
        ping -c1 -W1 "$1" &>/dev/null
    fi
}

# --- Python : aria_env ---
if [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
elif [[ -f aria_env/bin/activate ]]; then
    source aria_env/bin/activate
elif [[ -f .aria_env/bin/activate ]]; then
    source .aria_env/bin/activate
elif [[ -f ~/aria_env/bin/activate ]]; then
    source ~/aria_env/bin/activate
elif [[ -f ~/venv/bin/activate ]]; then
    source ~/venv/bin/activate
else
    echo "[run] WARNING: no virtual environment found, using system Python"
fi
PYTHON="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON" ]]; then
    echo "[run] ERROR: python3/python not found in PATH."
    exit 1
fi

# Ask the visitor for their preferences first (language, age, length).
# guide_setup.py writes .guide_profile.json, which main.py loads at the end.
# "$PYTHON" guide_setup.py


echo "[run] checking adb setup"
if ! command -v adb &>/dev/null; then
    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            echo "[run] adb not found, installing via brew..."
            brew install --cask android-platform-tools
        else
            echo "[run] ERROR: adb not found — install it with: brew install --cask android-platform-tools"
            exit 1
        fi
    else
        echo "[run] adb not found, installing..."
        sudo apt-get install -y adb
    fi
fi
if ! adb devices | grep -q 'device$'; then
    echo "[run] ADB kill and restart for clean run"
    adb kill-server
    sleep 1
    adb start-server
fi

if ! command -v aria &>/dev/null; then
    echo "[run] ERROR: 'aria' CLI not found in PATH. Install the Aria SDK and ensure 'aria' is on your PATH."
    exit 1
fi

# Need usb connection for this step
if ! aria auth check | grep -q authenticated; then
    echo "[run] Authenticate SDK with application if not already done. you have 10 seconds to approve on the app"
    aria auth pair
    sleep 10
fi

# Keep Wi-Fi alive on the glasses so the stream survives unplugging USB (idempotent, needs USB)
if [[ "$INTERFACE" == "wifi" ]]; then
    echo "[run] Enabling Wi-Fi keep-on on the device..."
    aria device wifi keep-on --set true || echo "[run] WARNING: could not set Wi-Fi keep-on (is USB connected?)"
fi

# Auto-detect Aria IP over WiFi if not provided
if [[ "$INTERFACE" == "wifi" && -z "$DEVICE_IP" ]]; then
    if ! command -v arp &>/dev/null; then
        if [[ "$OS" == "Darwin" ]]; then
            echo "[run] ERROR: 'arp' not found — it should be built into macOS, check /usr/sbin/arp."
            exit 1
        else
            echo "[run] net-tools not found, installing..."
            sudo apt-get install -y net-tools
        fi
    fi

    LAST_IP_FILE="$(dirname "$0")/.aria_last_ip"

    if [[ -f "$LAST_IP_FILE" ]]; then
        CACHED_IP=$(cat "$LAST_IP_FILE")
        if ping_once "$CACHED_IP"; then
            echo "[run] Aria reachable at cached IP: $CACHED_IP"
            DEVICE_IP="$CACHED_IP"
        else
            echo "[run] Cached IP $CACHED_IP unreachable, scanning ARP table..."
        fi
    fi

    if [[ -z "$DEVICE_IP" ]]; then
        DEVICE_IP=$(arp -a | grep -i '2c:26:17' \
                    | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if [[ -n "$DEVICE_IP" ]]; then
            echo "[run] Found Aria at $DEVICE_IP (ARP scan)"
            echo "$DEVICE_IP" > "$LAST_IP_FILE"
        else
            echo "[run] WARNING: Aria not found in ARP table — streaming without --device-ip"
        fi
    else
        echo "$DEVICE_IP" > "$LAST_IP_FILE"
    fi
fi

START_ARGS=(--interface "$INTERFACE" --profile "$PROFILE")
if [[ -n "$DEVICE_IP" ]]; then
    START_ARGS+=(--device-ip "$DEVICE_IP")
fi

if [[ ! -f "projectaria_client_sdk_samples/streaming_start.py" ]]; then
    echo "[run] ERROR: streaming_start.py not found. Check that projectaria_client_sdk_samples/ is present."
    exit 1
fi

echo "[run] Start streaming Aria (interface=$INTERFACE, profile=$PROFILE)..."
"$PYTHON" projectaria_client_sdk_samples/streaming_start.py "${START_ARGS[@]}"

STREAM_WAIT=10  # seconds to wait for the stream to initialise
echo "[run] Waiting ${STREAM_WAIT}s for stream to initialise..."
sleep "$STREAM_WAIT"

echo "[run] pipeline starting (main.py)..."
# Pass the device IP so main.py can stop streaming over Wi-Fi on Ctrl+C
# (USB/adb may be unplugged by then). Empty in USB mode -> falls back to adb.
export ARIA_DEVICE_IP="$DEVICE_IP"
exec "$PYTHON" main.py --profile-file .guide_profile.json