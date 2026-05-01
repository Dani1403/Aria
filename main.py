"""Museum Audio Guide - Sentence-by-sentence streaming pipeline.

Usage:
    python main.py <image_path>
    python main.py test_images/joconde.jpg
"""

import sys
import time
import queue
import shutil
import threading
from pathlib import Path

from dotenv import load_dotenv

from vision import stream_guide_sentences, stream_guide_sentences_from_bytes,STREAM_DONE
from tts import generate_sentence_audio
from audio import init_audio, play_audio_file, quit_audio, play_audio_bytes
from extract_frames import extract_frames_from_video
#from utils import pull_aria_recording
from stream import simulate_stream

from openai import OpenAI

def normalize_artwork(name: str) -> str:
    name = name.lower().strip()

    # remove punctuation & extra spaces
    import re
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name

#TODO : NOT GOOD
def is_similar_artwork(new_name: str, seen: set, threshold: float = 0.5) -> bool:
    """Check if new_name is similar to any name in the seen set.

    Uses word overlap ratio: if >= threshold of words match, it's a duplicate.
    E.g. "louvre pyramid" vs "glass pyramid" -> 1/2 = 50% -> duplicate.
    """
    new_words = set(new_name.split())
    for seen_name in seen:
        seen_words = set(seen_name.split())
        if not new_words or not seen_words:
            continue
        common = new_words & seen_words
        similarity = max(len(common) / len(new_words), len(common) / len(seen_words))
        if similarity >= threshold:
            return True
    return False




latency_start = {"t": None, 
                 "ux_done": False,
                 "real_done": False
                 }


def main(video_path: str, fps: float = 0.5):

    load_dotenv()

    if not Path(video_path).exists():
        print(f"Error: file not found -> {video_path}")
        return


    # Queues for communication between threads

    # Streaming: initialize VRS queue for new recordings
    vrs_q = queue.Queue()

    sentence_q = queue.Queue(maxsize=20)
    audio_q = queue.Queue()

    frame_q = queue.Queue()

    vision_error = []
    tts_error = []

    client = OpenAI()

    # Clean debug_frames folder at each run
    debug_dir = Path("debug_frames")
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir()



    # For pulling new VRS recordings, we can have a producer thread that checks for new recordings and puts their paths into a queue. It keeps track of seen paths to avoid duplicates.
    def vrs_worker():

        # while True:

        #     # Use any function that pulls the latest vrs recording from the streaming
        #     # The function should return the file path of the new recording, or None if no new recording is available
        #     # This is the heart of the streaming integration
        #    # vrs_path = pull_aria_recording(0)

        #     # for now, use two video files to simulate streaming by putting them into the queue with a delay
        #     vrs_path = video_path

        #     # Put the new VRS path into the queue
        #     if vrs_path:
        #         vrs_q.put(vrs_path)

        #     time.sleep(60)

        #for testing, use the function that simulates the stream
        simulate_stream(video, vrs_q, output_dir="stream_chunks",
                    chunk_duration=3.0, realtime=True)

    # Takes vrs file from the vrs queue and extracts frames, then puts them into the frame queue for processing by the vision worker.
    # for now extract from video files
    def frame_worker():
        while True:
            vrs_file = vrs_q.get()
            for idx, jpeg in extract_frames_from_video(vrs_file, fps):
                frame_q.put((idx, jpeg))

    # --- Thread 1: frame to vision ---
    def vision_worker():
        try:

            # pull from frame queue
            while True:
                idx, jpeg = frame_q.get()


                # Wait until TTS has caught up before processing a new frame
                while sentence_q.qsize() >= 5:
                    time.sleep(0.5)
                #attach a timestamp to measure latency to first audio
                timestamp = time.time()
                print(f"Processing frame {idx} with timestamp {timestamp:.2f}")
                with open(f"debug_frames/frame_{idx}.jpg", "wb") as f:
                    f.write(jpeg)
                try:
                    stream_guide_sentences_from_bytes(jpeg, timestamp, sentence_q, client)
                except Exception as e:
                    print(f"Error processing frame {idx}, skipping: {e}")
                    continue

        except Exception as e:
            vision_error.append(e)
        # finally:

        #     # BE CAREFUL to remove this when streaming
        #     sentence_q.put(STREAM_DONE)

    # --- Thread 2: TTS ---
    def tts_worker():

        #Start by generating the audio for the generic phrase
        generic_sentence = "This is a nice artwork let me tell you more about it !"
        generic_audio = generate_sentence_audio(
                        generic_sentence, client)

        try:
            seen_artworks = set()
            allow_description = False
            sentence_count = 0
            MAX_SENTENCES = 4

            while True:
                sentence, frame_timestamp = sentence_q.get()

                if sentence is STREAM_DONE:
                    break

                # -------------------------
                # HANDLE NONE
                # -------------------------
                if sentence.strip() == "NONE":
                    print("Got NONE skipping TTS.")
                    continue

                # -------------------------
                # ARTWORK HEADER
                # -------------------------
                if sentence.startswith("ARTWORK:"):




                    raw_name = sentence.replace("ARTWORK:", "").strip()
                    artwork_name = normalize_artwork(raw_name)

                    print(f"Detected artwork: {artwork_name}")

                    #check against ALL previously seen artworks
                    if is_similar_artwork(artwork_name, seen_artworks):
                        print(f"Similar artwork already seen, skipping: {artwork_name}")
                        allow_description = False
                        continue

                    # NEW artwork

                    # start timer when we detect a new artwork in order to measure time to first audio
                    latency_start["t"] = frame_timestamp
                    latency_start["ux_done"] = False
                    latency_start["real_done"] = False


                    seen_artworks.add(artwork_name)
                    allow_description = True
                    sentence_count = 0

                    print(f" NEW artwork {artwork_name}")

                    #play the generic phrase in order to fill the gap while the TTS is generating the first sentence
                    print(f"TTS: {generic_sentence}")
                    audio_q.put(("GENERIC", generic_audio))

                    continue  

                # -------------------------
                # DESCRIPTION SENTENCES
                # -------------------------
                if not allow_description:
                    continue

                if sentence_count >= MAX_SENTENCES:
                    continue

                print(f"TTS: {sentence}")

                audio_bytes = generate_sentence_audio(sentence, client)
                audio_q.put(("REAL", audio_bytes))

                sentence_count += 1

        except Exception as e:
            tts_error.append(e)

        finally:
            audio_q.put(STREAM_DONE)

    # Start threads

    #streaming: start vrs worker
    t_vrs = threading.Thread(target=vrs_worker)
    t_frame = threading.Thread(target=frame_worker)

    t_vision = threading.Thread(target=vision_worker)
    t_tts = threading.Thread(target=tts_worker)

    t_vision.start()
    t_tts.start()

    t_vrs.start()
    t_frame.start()


    # --- Main thread: playback ---
    init_audio()

    try:

        while True:

            audio_type, audio_bytes = audio_q.get()
            if audio_bytes is STREAM_DONE:
                break


            if latency_start["t"] is not None:

                if audio_type == "GENERIC" and not latency_start["ux_done"]:
                    elapsed = time.time() - latency_start["t"]
                    print(f"\n--- UX latency: {elapsed:.2f}s ---\n")
                    latency_start["ux_done"] = True

                if audio_type == "REAL" and not latency_start["real_done"]:
                    elapsed = time.time() - latency_start["t"]
                    print(f"\n--- REAL latency: {elapsed:.2f}s ---\n")
                    latency_start["real_done"] = True

                if latency_start["ux_done"] and latency_start["real_done"]:
                    latency_start["t"] = None
                    latency_start["ux_done"] = False
                    latency_start["real_done"] = False


            play_audio_bytes(audio_bytes)

    finally:
        quit_audio()

    t_vision.join()
    t_tts.join()

    if vision_error:
        print(f"Vision error: {vision_error[0]}")
    if tts_error:
        print(f"TTS error: {tts_error[0]}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [fps]")
        print("Example: python main.py Louvre2.mp4 0.5")
        sys.exit(1)

    video = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    main(video, fps=fps)
