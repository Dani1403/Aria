"""Walking detection from the Aria IMU.

Walking produces strong, periodic oscillations on the accelerometer, while
standing still leaves the magnitude close to gravity (~9.8 m/s2) with very
little variation. We therefore look at the standard deviation of the
accelerometer magnitude over a short sliding window: high std -> walking,
low std -> still. Hysteresis (two thresholds) plus a minimum dwell time avoid
flickering between the two states.
"""

import math
import time
import threading
from collections import deque


class WalkingDetector:
    """Sets / clears a threading.Event depending on whether the user walks.

    Args:
        walking_event: shared threading.Event. ``set()`` while walking.
        imu_idx: which IMU to listen to (Aria has two). We stick to one so we
            don't mix sensors running at different rates.
        window_sec: length of the sliding window used to compute the std.
        sample_rate: approximate IMU rate, used to size the window buffer.
        enter_threshold: std (m/s2) above which we declare "walking".
        exit_threshold: std (m/s2) below which we declare "still" again.
        min_state_time: minimum seconds to stay in a state before flipping.
    """

    def __init__(
        self,
        walking_event,
        imu_idx=1,
        window_sec=1.0,
        sample_rate=1000,
        enter_threshold=0.6,
        exit_threshold=0.3,
        min_state_time=0.4,
    ):
        self.walking_event = walking_event
        self.imu_idx = imu_idx
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_state_time = min_state_time

        self._mags = deque(maxlen=max(1, int(window_sec * sample_rate)))
        self._lock = threading.Lock()
        self._last_change = 0.0
        self._last_eval = 0.0

    def on_imu_received(self, samples, imu_idx):
        """Feed an IMU batch. Plug straight into the Aria observer callback."""
        if imu_idx != self.imu_idx:
            return
        with self._lock:
            for s in samples:
                ax, ay, az = s.accel_msec2
                self._mags.append(math.sqrt(ax * ax + ay * ay + az * az))

            # Re-evaluating the std on every batch (up to ~1 kHz) is wasteful;
            # 10 Hz is plenty for a gating decision.
            now = time.time()
            if now - self._last_eval < 0.1:
                return
            self._last_eval = now
            self._evaluate(now)

    def _evaluate(self, now):
        n = len(self._mags)
        if n < 10:
            return

        mean = sum(self._mags) / n
        std = math.sqrt(sum((m - mean) ** 2 for m in self._mags) / n)

        if now - self._last_change < self.min_state_time:
            return

        walking = self.walking_event.is_set()
        if not walking and std > self.enter_threshold:
            self.walking_event.set()
            self._last_change = now
            print(f"[IMU] walking -> audio paused (std={std:.2f})")
        elif walking and std < self.exit_threshold:
            self.walking_event.clear()
            self._last_change = now
            print(f"[IMU] still -> audio resumes (std={std:.2f})")
