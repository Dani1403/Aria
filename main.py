"""Museum Audio Guide - Sentence-by-sentence streaming pipeline.
"""
import os
import re
import sys
import time
from datetime import datetime
import queue
import shutil
import argparse
import threading
import subprocess
import random
from pathlib import Path

from dotenv import load_dotenv

from vision import stream_guide_sentences_from_bytes, STREAM_DONE, build_system_prompt
from tts import generate_sentence_audio
from audio import init_audio, play_audio_file, quit_audio, play_audio_bytes
from motion import WalkingDetector
from guide_profile import PROFILE_FILE, load_profile, cached_audio_path

#streaming
import cv2
import aria.sdk as aria
import numpy as np

from openai import OpenAI


def normalize_artwork(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name

# Words too generic to tell two artworks apart (EN/FR/ES articles and
# connectors). Without this, "portrait de louis xiv" and "portrait de
# napoleon" merge on "portrait de" and the second artwork is never described.
STOPWORDS = {
    "the", "a", "an", "of", "by", "in", "on", "and",
    "de", "du", "des", "le", "la", "les", "l", "d", "et",
    "el", "los", "las", "un", "une", "y",
}

#TODO : still word-based — a proper fix is visual matching (e.g. embeddings)
def is_similar_artwork(new_name: str, seen: dict, threshold: float = 0.6):
    """Check if new_name matches an already-seen artwork.

    Word overlap ratio on meaningful words only (stopwords removed), so
    "the mona lisa" and "mona lisa" match while two different portraits
    stay distinct.
    """
    new_words = set(new_name.split()) - STOPWORDS
    for seen_name in seen:
        seen_words = set(seen_name.split()) - STOPWORDS
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
    cmd = ["aria"]
    # Over Wi-Fi the USB/adb link may be unplugged, so the control connection
    # must reach the glasses by IP. run.sh exports ARIA_DEVICE_IP in that case.
    device_ip = os.environ.get("ARIA_DEVICE_IP")
    if device_ip:
        cmd += ["--device-ip", device_ip]
    cmd += ["streaming", "stop"]
    try:
        subprocess.run(cmd, check=False, timeout=30)
    except FileNotFoundError:
        print("[ARIA] 'aria' CLI not found in PATH, skipping streaming stop.")
    except subprocess.TimeoutExpired:
        print("[ARIA] streaming stop timed out.")
    except Exception as e:
        print(f"[ARIA] streaming stop failed: {e}")


def main():

    parser = argparse.ArgumentParser(description="Museum audio guide pipeline")
    parser.add_argument(
        "--profile-file",
        default=PROFILE_FILE,
        help="JSON visitor profile written by guide_setup.py",
    )
    args = parser.parse_args()

    load_dotenv()

    # Visitor profile: single source of truth for personalisation. Frozen for
    # the whole session; drives the vision prompt, the sentence cap and the
    # TTS voice/speed + localized framing phrases.
    profile = load_profile(args.profile_file)
    min_s, max_s = profile.sentence_range
    print(
        f"[PROFILE] language={profile.language_name} age={profile.age} ({profile.age_group}) "
        f"knowledge={profile.knowledge} length={profile.length} "
        f"({min_s}-{max_s} sentences, voice={profile.tts_voice}, speed={profile.tts_speed})"
    )
    system_prompt = build_system_prompt(profile)

    sentence_q = queue.Queue(maxsize=50)
    audio_q = queue.Queue()

    frame_q = queue.Queue()

    # Set while the user is walking (driven by the IMU). Used by tts_worker to
    # ignore vision input while moving.
    walking = threading.Event()

    # Gates playback. Set by the IMU detector the INSTANT walking starts (so it
    # never waits on a TTS call in flight) and released ONLY when the guide
    # (re)starts an artwork — never just because the user stopped moving. So
    # audio resumes on artwork detection, not on stop.
    paused = threading.Event()

    walking_detector = WalkingDetector(walking, pause_event=paused)

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
            try:
                image = np.rot90(image, -1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        def on_imu_received(self, samples, imu_idx):
            walking_detector.on_imu_received(samples, imu_idx)
        def on_streaming_client_failure(self, reason, message):
            print(f"[ARIA] Streaming Client Failure: {reason}: {message}")

    _aria_client = []  # holds the StreamingClient once subscribed, for clean shutdown

    def aria_worker():
        try:

            streaming_client = aria.StreamingClient()

            sub_config = streaming_client.subscription_config

            sub_config.subscriber_data_type = (
                aria.StreamingDataType.Rgb
                | aria.StreamingDataType.Imu
                # | aria.StreamingDataType.Slam
            )

            sub_config.message_queue_size[
                aria.StreamingDataType.Rgb
            ] = 1
            # Keep only the most recent IMU batch: we want a real-time walk/still
            # decision, not a backlog.
            sub_config.message_queue_size[
                aria.StreamingDataType.Imu
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

                # Stale frame: captured while walking, or before the last
                # walk -> still transition. Its guide would be dropped
                # downstream anyway, and a full stream takes 10-20s — skip the
                # API call entirely so a fresh frame is analysed sooner.
                if walking.is_set() or timestamp < walking_detector.last_still_time:
                    continue

                # Wait until TTS has caught up before processing a new frame
                while sentence_q.qsize() >= 10:
                    time.sleep(0.05)
                try:
                    stream_guide_sentences_from_bytes(jpeg, timestamp, sentence_q, client, profile.max_sentences, system_prompt)
                except Exception as e:
                    print(f"Error processing frame")
                    continue

        except Exception as e:
            vision_error.append(e)

    # --- Thread 3: TTS ---
    def tts_worker():

        print("[TTS] Worker started, waiting for sentences...")

        # Framing phrases in the profile's language (EN/FR/ES).
        OPENING_SENTENCES = profile.phrases["opening"]
        REENTRY_SENTENCES = profile.phrases["reentry"]
        ENDING_SENTENCES = profile.phrases["ending"]

        def tts(text):
            # Voice, speed and style directions come from the profile (age register).
            return generate_sentence_audio(
                text, client, profile.tts_voice, profile.tts_speed, profile.tts_instructions
            )

        def _load_cached_pool(pool_name, sentences):
            # Framing phrases are pregenerated once by pregenerate_audio.py and
            # committed to the repo (tts_cache/) — loaded from disk here so
            # startup never blocks on (or pays for) a live TTS call.
            pool = []
            for text in sentences:
                path = cached_audio_path(profile.language, profile.age_group, pool_name, text)
                if not path.exists():
                    print(
                        f"[TTS] MISSING cached audio for {pool_name} "
                        f"({profile.language}/{profile.age_group}): {text!r} — run pregenerate_audio.py"
                    )
                    continue
                pool.append((text, path.read_bytes()))
            return pool

        opening_audio_pool = _load_cached_pool("opening", OPENING_SENTENCES)
        reentry_audio_pool = _load_cached_pool("reentry", REENTRY_SENTENCES)
        ending_audio_pool = _load_cached_pool("ending", ENDING_SENTENCES)

        try:
            seen_artworks = dict()
            description_complete = set()
            allow_description = False
            sentence_count = 0

            in_artwork = False
            out_artwork = True
            current_artwork = None

            handled_walk_events = 0

            while True:
                # Walk starts are counted by the detector (which also sets
                # `paused` the instant they happen). Consuming the counter —
                # instead of comparing states — guarantees that even a short
                # walk gets its bookkeeping done, however long the TTS call
                # below kept us busy.
                walk_events = walking_detector.walk_events
                if walk_events != handled_walk_events:
                    handled_walk_events = walk_events
                    if in_artwork:
                        print("[TTS] Walking started: finishing current sentence, then pausing.")
                        # Save everything still queued so this artwork can be
                        # resumed later, right where we left off. The sentence
                        # currently playing has already left the queue, so it
                        # finishes normally (no abrupt cut). Playback is already
                        # gated by `paused`, so nothing races us here.
                        while not audio_q.empty():
                            try:
                                item = audio_q.get_nowait()
                                audio_type, _ = item
                                if audio_type not in ("GENERIC", "REENTRY", "NAME"):
                                    seen_artworks[current_artwork].append(item)
                            except queue.Empty:
                                break
                        in_artwork = False
                        out_artwork = True
                        allow_description = False
                    paused.set()

                try:
                    sentence, frame_timestamp = sentence_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if sentence is STREAM_DONE:
                    break

                # Only act on guides of frames seen while standing still: drop
                # everything while walking, and everything from frames captured
                # before the last walk -> still transition (stale streams whose
                # ARTWORK header may already have been dropped mid-walk).
                if walking.is_set() or frame_timestamp < walking_detector.last_still_time:
                    continue

                # -------------------------
                # HANDLE END — the guide of the current artwork is finished:
                # nothing will ever be regenerated for it, and it no longer
                # blocks switching to another artwork without walking.
                # -------------------------
                if sentence.strip().rstrip('.!?') == "END":
                    if in_artwork and sentence_count > 0 and current_artwork not in description_complete:
                        description_complete.add(current_artwork)
                        allow_description = False
                        if ending_audio_pool:
                            text, audio_bytes = random.choice(ending_audio_pool)
                            print(f"[TTS] END: {text}")
                            audio_q.put(("END", audio_bytes))
                    continue

                # -------------------------
                # HANDLE NONE  (no longer pauses — movement is the pause trigger)
                # -------------------------
                if sentence.strip() == "NONE":
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

                    if in_artwork:
                        if similar == current_artwork:
                            # Same artwork re-detected while we are on it: its
                            # guide already ran or is running, never restart it.
                            allow_description = False
                            continue
                        if current_artwork not in description_complete:
                            # Another artwork glimpsed WITHOUT walking while the
                            # current guide is not finished: ignore it. Walking
                            # (or finishing the guide) is what allows switching.
                            print(f"[TTS] ignoring '{artwork_name}' — current guide not finished")
                            continue
                        # Current guide is finished: switching without walking
                        # is allowed.
                        in_artwork = False
                        out_artwork = True

                    if similar is not None:
                        # Already-seen artwork: replay whatever was left
                        # unheard, never regenerate (the guide is written once).
                        in_artwork = True
                        out_artwork = False
                        current_artwork = similar
                        allow_description = False
                        sentence_count = 0

                        saved = seen_artworks[similar]
                        seen_artworks[similar] = []
                        # Replay-only resume: after this, nothing is ever added.
                        description_complete.add(similar)

                        if saved:
                            # Greet only when there is audio left to play.
                            if reentry_audio_pool:
                                text, audio_bytes = random.choice(reentry_audio_pool)
                                print(f"[TTS] REENTRY: {text}")
                                audio_q.put(("REENTRY", audio_bytes))
                            for s in saved:
                                audio_q.put(s)
                        else:
                            print(f"[TTS] guide for '{similar}' already fully heard — staying silent")

                        # Resume playback.
                        paused.clear()
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

                    print(f" NEW artwork: {artwork_name}")

                    #play the generic phrase in order to fill the gap while the TTS is generating the first sentence
                    if opening_audio_pool:
                        text, audio_bytes = random.choice(opening_audio_pool)
                        print(f"[TTS] GENERIC: {text}")
                        audio_q.put(("GENERIC", audio_bytes))

                    # Release playback BEFORE generating the name clip, so the
                    # generic phrase masks the TTS latency (paused is set on
                    # every walk now, so waiting here would delay the first
                    # sound by a full TTS call).
                    paused.clear()

                    # Announce the artwork name at the start of the description.
                    # The header "ARTWORK: <name>" is otherwise only used for
                    # state, so without this the name is never spoken. Use
                    # raw_name to keep casing.
                    if raw_name:
                        try:
                            name_text = profile.phrases["name"].format(name=raw_name)
                            name_audio = tts(name_text)
                            print(f"[TTS] NAME: {name_text}")
                            audio_q.put(("NAME", name_audio))
                        except Exception as e:
                            print(f"[TTS] Name announcement failed: {e}")

                    continue

                # -------------------------
                # DESCRIPTION SENTENCES
                # -------------------------
                if not allow_description:
                    continue

                # Length preset cap — same limit as the vision side, otherwise
                # extra sentences would be silently dropped here.
                if sentence_count >= profile.max_sentences:
                    continue

                print(f"[TTS] REAL: {sentence}")

                audio_bytes = tts(sentence)
                # If walking started while this sentence was being generated,
                # it must not play now: stash it for the resume instead. (The
                # transition handler at the top of the loop only drains what is
                # already in audio_q — this clip isn't there yet.)
                if walking.is_set() or paused.is_set():
                    seen_artworks[current_artwork].append(("REAL", audio_bytes))
                else:
                    audio_q.put(("REAL", audio_bytes))

                sentence_count += 1

        except Exception as e:
            tts_error.append(e)

        finally:
            audio_q.put((STREAM_DONE, STREAM_DONE))

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

    #initialize session audio for montage
    session_audio_dir = Path("session_audio")
    session_audio_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(f"session_audio/{datetime.now().strftime('%Y_%m_%d_%H_%M')}")
    session_dir.mkdir(parents=True, exist_ok=True)

    #copy guide profile inside session for reference
    shutil.copy2(PROFILE_FILE, session_dir)

    interrupted = False
    try:
        while True:
            # Hold here while paused. Pausing is triggered by walking and is
            # released only when the guide (re)starts an artwork — never just by
            # standing still — so audio never restarts on stop alone. The
            # sentence already playing is not interrupted: this gate is checked
            # before pulling the next one.
            while paused.is_set():
                time.sleep(0.05)

            # Timeout so the pause gate above is re-checked periodically even
            # while the queue is empty (a blocking get would let one clip slip
            # through if the pause landed while we were waiting).
            try:
                audio_type, audio_bytes = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
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

            #for video montage
            audio_timestamp = time.time()

            # Play the whole sentence to the end (never cut mid-sentence).
            play_audio_bytes(audio_bytes)

            #add the audio to the session audio directory for video montage
            with open(f"{session_dir}/audio_{audio_timestamp}_{audio_type}.mp3", "wb") as f:
                f.write(audio_bytes)

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