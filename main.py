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
from audio import init_audio, play_audio_file, quit_audio
from extract_frames import extract_frames, extract_frames_from_video
from openai import OpenAI


def prev_main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        print("Example: python main.py test_images/joconde.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: image not found -> {image_path}")
        sys.exit(1)

    start_time = time.time()

    # Queue: vision pushes sentences, main thread consumes them
    sentence_q = queue.Queue()

    # Queue: TTS worker pushes ready MP3 paths, playback thread consumes them
    audio_q = queue.Queue()

    vision_error = []
    tts_error = []

    # --- Thread 1: Vision streams sentences into sentence_q ---
    def vision_worker():
        try:
            print(f"Analyzing image: {image_path}")
            stream_guide_sentences(image_path, sentence_q)
        except Exception as e:
            vision_error.append(e)
            sentence_q.put(STREAM_DONE)

    # --- Thread 2: TTS consumes sentences, produces MP3 files ---
    def tts_worker():
        idx = 0
        try:
            while True:
                sentence = sentence_q.get()
                if sentence is STREAM_DONE:
                    break
                print(f"  TTS: \"{sentence}\"")
                mp3_path = f"sentence_{idx}.mp3"
                generate_sentence_audio(sentence, mp3_path)
                audio_q.put(mp3_path)
                idx += 1
        except Exception as e:
            tts_error.append(e)
        finally:
            audio_q.put(STREAM_DONE)

    # Start vision and TTS threads
    t_vision = threading.Thread(target=vision_worker)
    t_tts = threading.Thread(target=tts_worker)
    t_vision.start()
    t_tts.start()

    # --- Main thread: play MP3 files as they arrive ---
    init_audio()
    try:
        first = True
        while True:
            mp3_path = audio_q.get()
            if mp3_path is STREAM_DONE:
                break
            if first:
                elapsed = time.time() - start_time
                print(f"\n--- Time to first audio: {elapsed:.2f}s ---")
                print("Playing audio guide...\n")
                first = False
            play_audio_file(mp3_path)
            # Clean up the temporary file
            Path(mp3_path).unlink(missing_ok=True)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Playback error: {e}")
    finally:
        quit_audio()

    # Wait for threads to finish
    t_vision.join()
    t_tts.join()

    if vision_error:
        print(f"Vision error: {vision_error[0]}")
        sys.exit(1)
    if tts_error:
        print(f"TTS error: {tts_error[0]}")
        sys.exit(1)

    print("\nPlayback complete.")



def normalize_artwork(name: str) -> str:
    name = name.lower().strip()

    # remove punctuation & extra spaces
    import re
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name


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


def main(video_path: str, fps: float = 0.5):

    load_dotenv()

    if not Path(video_path).exists():
        print(f"Error: file not found -> {video_path}")
        return

    start_time = time.time()

    sentence_q = queue.Queue(maxsize=20)
    audio_q = queue.Queue()

    vision_error = []
    tts_error = []

    client = OpenAI()

    # Clean debug_frames folder at each run
    debug_dir = Path("debug_frames")
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir()

    # --- Thread 1: frame to vision ---
    def vision_worker():
        try:
            for idx, jpeg in extract_frames_from_video(video_path, target_fps=fps):
                # Wait until TTS has caught up before processing a new frame
                while sentence_q.qsize() >= 5:
                    time.sleep(0.5)
                print(f"\n--- Frame {idx} ---\n")
                with open(f"debug_frames/frame_{idx}.jpg", "wb") as f:
                    f.write(jpeg)
                try:
                    stream_guide_sentences_from_bytes(jpeg, sentence_q, client)
                except Exception as e:
                    print(f"Error processing frame {idx}, skipping: {e}")
                    continue

        except Exception as e:
            vision_error.append(e)
        finally:
            sentence_q.put(STREAM_DONE)

    # --- Thread 2: TTS ---
    def tts_worker():
        idx = 0
        try:
            seen_artworks = set()
            allow_description = False
            sentence_count = 0
            MAX_SENTENCES = 4

            while True:
                sentence = sentence_q.get()

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

                    # fuzzy check against ALL previously seen artworks
                    if is_similar_artwork(artwork_name, seen_artworks):
                        print(f"Similar artwork already seen, skipping: {artwork_name}")
                        allow_description = False
                        continue

                    # NEW artwork
                    seen_artworks.add(artwork_name)
                    allow_description = True
                    sentence_count = 0

                    print(f" NEW artwork {artwork_name}")

                    # speak header
                    print(f"TTS: {sentence}")
                    mp3_path = f"sentence_{idx}.mp3"
                    generate_sentence_audio(sentence, mp3_path, client)
                    audio_q.put(mp3_path)
                    idx += 1

                    continue  # IMPORTANT

                # -------------------------
                # DESCRIPTION SENTENCES
                # -------------------------
                if not allow_description:
                    continue

                if sentence_count >= MAX_SENTENCES:
                    continue

                print(f"TTS: {sentence}")

                mp3_path = f"sentence_{idx}.mp3"
                generate_sentence_audio(sentence, mp3_path, client)
                audio_q.put(mp3_path)
                idx += 1

                sentence_count += 1

        except Exception as e:
            tts_error.append(e)

        finally:
            audio_q.put(STREAM_DONE)

    # Start threads
    t_vision = threading.Thread(target=vision_worker)
    t_tts = threading.Thread(target=tts_worker)

    t_vision.start()
    t_tts.start()

    # --- Main thread: playback ---
    init_audio()

    try:
        first = True

        while True:
            mp3_path = audio_q.get()

            if mp3_path is STREAM_DONE:
                break

            if first:
                elapsed = time.time() - start_time
                print(f"\n--- Time to first audio: {elapsed:.2f}s ---\n")
                first = False

            play_audio_file(mp3_path)
            Path(mp3_path).unlink(missing_ok=True)

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