"""Museum Audio Guide - Sentence-by-sentence streaming pipeline.
"""
import sys
import time
import queue
import shutil
import threading
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from vision import stream_guide_sentences, stream_guide_sentences_from_bytes,STREAM_DONE
from tts import generate_sentence_audio
from audio import init_audio, play_audio_file, quit_audio, play_audio_bytes
from extract_frames import extract_frames_from_video
#from utils import pull_aria_recording
#from stream import simulate_stream

#streaming
import cv2
import aria.sdk as aria
import numpy as np

from visualizer import BaseStreamingClientObserver

from openai import OpenAI

def normalize_artwork(name: str) -> str:
    name = name.lower().strip()

    # remove punctuation & extra spaces
    import re
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name

#TODO : NOT GOOD
def is_similar_artwork(new_name: str, seen: dict, threshold: float = 0.5):
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
            return seen_name
    return None

latency_start = {"t": None,
                 "ux_done": False,
                 "real_done": False
                 }


def stop_aria_streaming():
    try:
        subprocess.run(["aria", "streaming", "stop"], check=False, timeout=30)
    except FileNotFoundError:
        print("[ARIA] 'aria' CLI not found in PATH, skipping streaming stop.")
    except subprocess.TimeoutExpired:
        print("[ARIA] streaming stop timed out.")
    except Exception as e:
        print(f"[ARIA] streaming stop failed: {e}")


def main(video_path: str = None, fps: float = 0.5):

    load_dotenv()

    sentence_q = queue.Queue(maxsize=50)
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

    class AriaObserver():
        def __init__(self):
            pass
        def on_image_received(self, image, record):
            #print("[ARIA] IMAGE CALLBACK")
            try:
                image = np.rot90(image, -1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # if record.camera_id != aria.CameraId.EyeTrack:
                #     image = np.rot90(image)
                # else:
                #     image = np.rot90(image,2)
                success, jpeg = cv2.imencode(
                    ".jpg",
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY),70]
                )
                if not success:
                    return
                if frame_q.qsize() >= 1:
                    return
                frame_q.put(
                    (time.time(), jpeg.tobytes()),
                    block=False
                )
                timestamp = time.time()
                print(f"Processing frame {timestamp} with timestamp {timestamp:.2f}")
                with open(f"debug_frames/frame_{timestamp}.jpg", "wb") as f:
                    f.write(jpeg)

                print(
                    f"[ARIA] queued q={frame_q.qsize()}"
                )
            except Exception as e:
                print(e)
        def on_streaming_client_failure(self, reason, message):
            print(f"[ARIA] Streaming Client Failure: {reason}: {message}")

    _aria_client = []  # holds the StreamingClient once subscribed, for clean shutdown

    def aria_worker():
        try:

            streaming_client = aria.StreamingClient()

            sub_config = streaming_client.subscription_config

            sub_config.subscriber_data_type = (
                aria.StreamingDataType.Rgb 
                # | aria.StreamingDataType.Slam
                # | aria.StreamingDataType.Imu
            )

            sub_config.message_queue_size[
                aria.StreamingDataType.Rgb
            ] = 1

            options = aria.StreamingSecurityOptions()
            options.use_ephemeral_certs = True
            sub_config.security_options = options

            streaming_client.subscription_config = sub_config

            print(
                "[ARIA] Subscription data type:",   
                streaming_client.subscription_config.subscriber_data_type
            )

            observer = AriaObserver()

            streaming_client.set_streaming_client_observer(
                observer
            )

            print("[ARIA] Subscribing...")

            streaming_client.subscribe()
            _aria_client.append(streaming_client)

            print("[ARIA] subscribed?", streaming_client.is_subscribed())

            while True:
                cv2.waitKey(1)
                time.sleep(0.01)

        except Exception as e:
            print(f"[ARIA] Worker fatal error: {e}")

    # --- Thread 2: frame to vision ---
    def vision_worker():
        try:

            print("[VISION] Worker started, waiting for frames...")

            # pull from frame queue
            while True:
                timestamp, jpeg = frame_q.get()


                # Wait until TTS has caught up before processing a new frame
                while sentence_q.qsize() >= 10:
                    time.sleep(0.5)
                #attach a timestamp to measure latency to first audio
                timestamp = time.time()
                # print(f"Processing frame {timestamp} with timestamp {timestamp:.2f}")
                # with open(f"debug_frames/frame_{timestamp}.jpg", "wb") as f:
                #     f.write(jpeg)
                try:
                    stream_guide_sentences_from_bytes(jpeg, timestamp, sentence_q, client)
                except Exception as e:
                    #print(f"Error processing frame {idx}, skipping: {e}")
                    print(f"Error processing frame")
                    continue

        except Exception as e:
            vision_error.append(e)

    # --- Thread 3: TTS ---
    def tts_worker():

        print("[TTS] Worker started, waiting for sentences...")

        #Start by generating the audio for the generic phrase
        generic_sentence = "This is a nice artwork let me tell you more about it !"
        generic_audio = generate_sentence_audio(
                        generic_sentence, client)

        #TODO : vary generic sentences.

        try:
            seen_artworks = dict()
            allow_description = False
            sentence_count = 0
            MAX_SENTENCES = 4

            in_artwork = False
            out_artwork = True
            current_artwork = None

            while True:
                sentence, frame_timestamp = sentence_q.get()

                if sentence is STREAM_DONE:
                    break

                # -------------------------
                # HANDLE NONE
                # -------------------------
                if sentence.strip() == "NONE":
                    print(f"Got NONE on frame with timestamp {frame_timestamp} skipping TTS.")


                    if in_artwork:
                        print("End of artwork description.")

                        # empty audio queue to stop any pending audio from playing after the artwork is gone
                        while not audio_q.empty():
                            try:
                                #audio_q.get_nowait()
                                seen_artworks[current_artwork].append(
                                    audio_q.get_nowait())
                            except queue.Empty:
                                break

                    in_artwork = False
                    out_artwork = True
                    allow_description = False
                    continue

                # -------------------------
                # ARTWORK HEADER
                # -------------------------
                if sentence.startswith("ARTWORK:"):

                    raw_name = sentence.replace("ARTWORK:", "").strip()
                    artwork_name = normalize_artwork(raw_name)

                    print(f"Detected artwork: {artwork_name}")

                    #check against ALL previously seen artworks
                    similar = is_similar_artwork(artwork_name, seen_artworks)
                    if similar is not None:

                        if out_artwork:

                            #We were out of artwork and now we re enter it.
                            #We can then continue explaining

                            #TODO: Add generic sentence like "welcome back" with name of artwork

                            in_artwork = True
                            out_artwork = False
                            allow_description = True   
                            sentence_count = 0

                            current_artwork = similar

                            saved = seen_artworks[similar]
                            for s in saved:
                                audio_q.put(s)

                            seen_artworks[similar] = []

                            continue


                        else:
                            print(f"Similar artwork already seen, skipping: {artwork_name}")
                            allow_description = False
                            continue

                    # NEW artwork

                    # start timer when we detect a new artwork in order to measure time to first audio
                    latency_start["t"] = frame_timestamp
                    latency_start["ux_done"] = False
                    latency_start["real_done"] = False

                    seen_artworks[artwork_name] = []
                    allow_description = True
                    sentence_count = 0
                   
                    in_artwork = True
                    out_artwork = False

                    current_artwork = artwork_name

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


    #streaming: start aria worker
    t_aria = threading.Thread(
        target=aria_worker,
        daemon=True
    )
    t_aria.start()

    t_vision = threading.Thread(target=vision_worker, daemon=True)
    t_tts = threading.Thread(target=tts_worker, daemon=True)

    t_vision.start()
    t_tts.start()


    # --- Main thread: playback ---
    init_audio()

    interrupted = False
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

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted, shutting down...")
        interrupted = True

    finally:
        quit_audio()
        if _aria_client:
            try:
                _aria_client[0].unsubscribe()
                print("[ARIA] Unsubscribed cleanly.")
            except Exception as e:
                print(f"[ARIA] unsubscribe() failed: {e}")
        stop_aria_streaming()

    if not interrupted:
        t_vision.join()
        t_tts.join()

        if vision_error:
            print(f"Vision error: {vision_error[0]}")
        if tts_error:
            print(f"TTS error: {tts_error[0]}")

        print("\nPipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted during init, shutting down...")
        stop_aria_streaming()
        sys.exit(0)
