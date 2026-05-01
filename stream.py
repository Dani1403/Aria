import cv2
import time
from pathlib import Path


def simulate_stream(
    video_path: str,
    queue,
    output_dir: str = "stream_chunks",
    chunk_duration: float = 3.0,   
    realtime: bool = True,
):


    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Impossible to open")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    frames_per_chunk = int(fps * chunk_duration)

    print(f"[STREAM] FPS: {fps:.2f}")
    print(f"[STREAM] Frames per chunk: {frames_per_chunk}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    chunk_idx = 0
    frame_idx = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    while True:

        chunk_file = output_path / f"chunk_{chunk_idx:04d}.mp4"

        writer = None
        frames_written = 0

        start_chunk_time = time.time()

        while frames_written < frames_per_chunk:
            ret, frame = cap.read()
            if not ret:
                break

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(chunk_file),
                    fourcc,
                    fps,
                    (w, h)
                )

            writer.write(frame)

            frames_written += 1
            frame_idx += 1

        if writer:
            writer.release()

        if frames_written == 0:
            break

        print(f"[STREAM] Wrote {chunk_file}")

        queue.put(str(chunk_file))
        print(f"[STREAM] Enqueued {chunk_file}")

        if realtime:
            elapsed = time.time() - start_chunk_time
            sleep_time = max(0, chunk_duration - elapsed)
            time.sleep(sleep_time)

        chunk_idx += 1

    cap.release()

    print("[STREAM] Streaming over")