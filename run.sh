#!/usr/bin/env bash
#
# Run the audioguide from start to finish :
#   1. start streaming (streaming_start.py)
#   2. run pipeline (main.py) that connects to the active flux
#
# Usage :
#   ./run.sh                             # USB, profile18 (default)
#   ./run.sh wifi                        # WiFi — IP auto-detected via ARP (requires net-tools)
#   ./run.sh wifi profile18              # WiFi + explicit profile, IP auto-detected
#   ./run.sh wifi profile18 172.20.10.2  # WiFi + explicit IP, skips ARP scan
#
# Note: ARP auto-detection requires net-tools (arp). Installed automatically if missing.
#
set -euo pipefail

cd "$(dirname "$0")"

OS="$(uname -s)"   # Linux or Darwin (macOS)
INTERFACE="${1:-usb}"
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
if [[ -f aria_env/bin/activate ]]; then
    source aria_env/bin/activate
elif [[ -f .aria_env/bin/activate ]]; then
    source .aria_env/bin/activate
elif [[ -f ~/aria_env/bin/activate ]]; then
    source ~/aria_env/bin/activate
fi
PYTHON="$(command -v python3 || command -v python)"

if ! adb devices | grep -q 'device$'; then
    echo "[run] ADB kill and restart for clean run"
    adb kill-server
    sleep 1
    adb start-server
fi

# Need usb connection for this step
if ! aria auth check | grep -q authenticated; then
    echo "[run] Authenticate SDK with application if not already done. you have 10 seconds to approve on the app"
    aria auth pair
    sleep 10
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

echo "[run] Start streaming Aria (interface=$INTERFACE, profile=$PROFILE)..."
"$PYTHON" projectaria_client_sdk_samples/streaming_start.py "${START_ARGS[@]}"

echo "[run] Wait for flux init ..."
sleep 10

echo "[run] pipeline starting (main.py)..."
exec "$PYTHON" main.py
