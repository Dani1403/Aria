"""Live IMU readout for the Aria glasses.

Use it to (1) confirm IMU data actually arrives and (2) tune the walk/still
thresholds in motion.py.

Prerequisite: streaming must already be running. Either run the normal flow
(./run.sh starts streaming) and launch this *instead of / alongside* main.py,
or start streaming by hand:

    ./venv/bin/python projectaria_client_sdk_samples/streaming_start.py --interface usb
    # (or: --interface wifi --device-ip <ip>)

Then:

    ./venv/bin/python imu_debug.py

Walk around, then stand still, and watch the `std` column for each IMU:
it should sit near ~0 when still and climb clearly when you walk. Pick
enter/exit thresholds in motion.py from those two regimes. Ctrl+C to stop.
"""

import math
import time
from collections import deque

import aria.sdk as aria

WINDOW = 1000  # samples kept per IMU (~1 s at 1 kHz)


def main():
    aria.set_log_level(aria.Level.Info)

    client = aria.StreamingClient()
    config = client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Imu
    config.message_queue_size[aria.StreamingDataType.Imu] = 1
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    client.subscription_config = config

    mags = {0: deque(maxlen=WINDOW), 1: deque(maxlen=WINDOW)}
    counts = {0: 0, 1: 0}

    class Observer:
        def on_imu_received(self, samples, imu_idx):
            buf = mags.setdefault(imu_idx, deque(maxlen=WINDOW))
            for s in samples:
                ax, ay, az = s.accel_msec2
                buf.append(math.sqrt(ax * ax + ay * ay + az * az))
            counts[imu_idx] = counts.get(imu_idx, 0) + len(samples)

        def on_streaming_client_failure(self, reason, message):
            print(f"[IMU] streaming failure {reason}: {message}")

    client.set_streaming_client_observer(Observer())
    print("[IMU] subscribing...  (Ctrl+C to stop)")
    client.subscribe()

    try:
        while True:
            time.sleep(0.2)
            parts = []
            for idx in (0, 1):
                buf = mags[idx]
                if buf:
                    n = len(buf)
                    mean = sum(buf) / n
                    std = math.sqrt(sum((m - mean) ** 2 for m in buf) / n)
                    parts.append(
                        f"imu{idx}: |a|={mean:5.2f}  std={std:5.2f}  n={counts[idx]:7d}"
                    )
                else:
                    parts.append(f"imu{idx}: ---- no data ----")
            print("   |   ".join(parts))
    except KeyboardInterrupt:
        pass
    finally:
        client.unsubscribe()
        print("\n[IMU] unsubscribed.")


if __name__ == "__main__":
    main()
