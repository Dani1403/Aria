try:
    from pyvrs.reader import SyncVRSReader
except ImportError:
    SyncVRSReader = None

import math
import cv2

def estimate_fps(reader, stream_id, frames):
    """
    Estimate FPS using timestamps.
    """
    if len(frames) < 2:
        return 1.0

    ts0 = frames[0].timestamp
    ts1 = frames[-1].timestamp

    duration = ts1 - ts0

    if duration <= 0:
        return 1.0

    return (len(frames) - 1) / duration


def extract_frames(vrs_file, target_fps=2):

    reader = SyncVRSReader(vrs_file)

    stream_id = "214-1"

    frames = reader.filtered_by_fields(
        stream_ids={stream_id},
        record_types={"data"}
    )

    if len(frames) == 0:
        reader.close()
        return

    # estimate original FPS
    original_fps = estimate_fps(reader, stream_id, frames)

    print(f"Original FPS: {original_fps:.2f}")

    step = max(1, int(round(original_fps / target_fps)))

    print(f"Sampling every {step} frames to reach ~{target_fps} FPS")

    for i in range(0, len(frames), step):

        rec = frames[i]

        if len(rec.image_blocks) == 0:
            continue

        jpeg_bytes = bytes(rec.image_blocks[0])

        yield i, jpeg_bytes

    reader.close()

def extract_frames_from_video(video_path, target_fps=1.0):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: cannot open video")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 25  # fallback

    print(f"Original FPS: {original_fps:.2f}")

    step = max(1, int(round(original_fps / target_fps)))
    print(f"Sampling every {step} frames")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # encode frame to JPEG (like VRS)
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                yield frame_idx, buffer.tobytes()

        frame_idx += 1

    cap.release()