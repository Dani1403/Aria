#!/usr/bin/env python3
import argparse, sys, json, math
from pathlib import Path

import numpy as np
import cv2

# pyvrs is installed as package name "vrs"
from pyvrs.reader import SyncVRSReader

def pick_image_stream(reader, prefer=None):
    """Pick a stream that contains images. If 'prefer' is given, return it if valid."""
    if prefer and prefer in reader.stream_ids and reader.might_contain_images(prefer):
        return prefer
    # Otherwise, pick the first stream that might contain images.
    for sid in sorted(reader.stream_ids):
        try:
            if reader.might_contain_images(sid):
                return sid
        except Exception:
            pass
    raise RuntimeError("No image streams found in this VRS.")

def guess_fps(reader, stream_id, indices):
    # Try pyvrs' estimate first
    try:
        fps = float(reader.get_estimated_frame_rate(stream_id))
        if math.isfinite(fps) and fps > 0:
            return fps
    except Exception:
        pass
    # Fallback: estimate from timestamps of first/last
    try:
        ts = reader.get_timestamp_list(indices)
        if len(ts) >= 2:
            duration = ts[-1] - ts[0]
            if duration > 0:
                return (len(ts) - 1) / duration
    except Exception:
        pass
    return 30.0  # sensible default

def decode_block_to_bgr(spec, block_bytes):
    arr = np.frombuffer(block_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    raise RuntimeError("Unknown image block type; could not decode.")

def VRSToVideo(vrsdir):
    if vrsdir.is_file():
        paths = [vrsdir]
    else:

        paths = [
            p for p in vrsdir.iterdir()
            if p.suffix == ".vrs" and not (p.with_suffix('.mp4').exists())
            ]

    for vrs_path in paths:
        print(f"[VRS ] Processing {vrs_path}")
        reader = SyncVRSReader(str(vrs_path))
        try:
            stream_id = pick_image_stream(reader, '214-1')
            print(f"[VRS ] Using stream: {stream_id}")

            # Build an iterator over data records for this stream
            fr = reader.filtered_by_fields(stream_ids={stream_id}, record_types={"data"})

            # Collect indices for FPS estimation and to count frames
            indices = list(range(len(fr)))
            if len(indices) == 0:
                print("[VRS ] No data records found for that stream.")
                continue

            fps = guess_fps(reader, stream_id, indices)
            print(f"[VRS ] FPS: {fps:.3f}")

            # Prime first frame to get frame size
            first = fr[0]
            if len(first.image_blocks) == 0:
                print("[VRS ] First data record has no image blocks.")
                continue

            # decode first block of first record
            first_bgr = None
            for spec, block in zip(first.image_specs, first.image_blocks):
                block_bytes = memoryview(block)
                with open("test.jpg", "wb") as f:
                    f.write(bytes(block))
                first_bgr = decode_block_to_bgr(spec, block_bytes)
                if first_bgr is not None:
                    break

            if first_bgr is None:
                print("[VRS ] Could not decode the first image frame.")
                continue
                # Set output_video name to match input, but with .mp4 extension
            output_video = vrs_path.with_suffix('.mp4')
            h, w = first_bgr.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h), isColor=True)
            if not writer.isOpened():
                print("[VRS ] Failed to open VideoWriter. Try a different codec or filename (.avi).")
                continue

            # write first
            first_bgr = cv2.rotate(first_bgr, cv2.ROTATE_90_CLOCKWISE) # Because the image comes pre-rotated
            writer.write(first_bgr)

            frames = 1
            first_timestamp = fr[0].timestamp
            for rec in fr[1:]:
                video_time = rec.timestamp - first_timestamp
                if len(rec.image_blocks) == 0:
                    continue
                spec, block = rec.image_specs[0], rec.image_blocks[0] # Use first block to get image specs
                try:
                    bgr = decode_block_to_bgr(spec, memoryview(block))
                    if bgr.shape[0] != h or bgr.shape[1] != w:
                        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                    writer.write(bgr)
                    frames += 1
                except Exception as e:
                    print(f"[VRS ] Skipping frame (decode error): {e}")

            writer.release()
            print(f"[VRS ] Wrote {frames} frames to {output_video}")

        finally:
            reader.close()

if __name__ == "__main__":
    # VRSToVideo(Path('/home/reggev/shared/aria'))  # Example usage
    VRSToVideo(Path('/home/daniel/shared_aria/Stream.vrs')),


