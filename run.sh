#!/usr/bin/env bash
#
# Lance l'audioguide Aria de bout en bout :
#   1. start streaming (streaming_start.py)
#   2. run pipeline (main.py) that connects to the active flux
#
# Usage :
#   ./run.sh                  # interface USB, profile18 (default)
#   ./run.sh wifi             # interface wifi
#   ./run.sh wifi profile18 192.168.1.42   # wifi + IP of device
#
set -euo pipefail

cd "$(dirname "$0")"

INTERFACE="${1:-usb}"
PROFILE="${2:-profile18}"
DEVICE_IP="${3:-}"

# --- Python : aria_env ---
if [[ -f aria_env/bin/activate ]]; then
    source aria_env/bin/activate
elif [[ -f .aria_env/bin/activate ]]; then
    source .aria_env/bin/activate
elif [[ -f ~/aria_env/bin/activate ]]; then
    source ~/aria_env/bin/activate
fi
PYTHON="$(command -v python3 || command -v python)"

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
