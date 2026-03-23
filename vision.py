"""Artwork recognition module using GPT-4o Vision with streaming."""

import base64
import io
import queue
import re
from pathlib import Path

from PIL import Image
from openai import OpenAI, APIError, APIConnectionError
import base64

SYSTEM_PROMPT = (
    "You are a museum audio guide in an augmented reality system.\n\n"

    "Your task:\n"
    "- If you see something that looks like an artwork (painting, sculpture, installation), "
    "describe it as a museum guide.\n"
    "- The artwork may be small, partially visible, behind glass, or in a crowded scene.\n\n"

    "IMPORTANT:\n"
    "- It does NOT need to be perfectly identified\n"
    "- If it looks like an artwork, assume it is one and describe it\n"
    "- Prefer describing rather than missing\n\n"

    "CRITICAL RULE:\n"
    "- If there is clearly NO artwork, output exactly: NONE\n"
    "- Output ONLY the word NONE (no punctuation, no explanation)\n\n"

    "If there is an artwork:\n"
    "- Provide a concise but rich explanation (3 to 5 sentences)\n"
    "- The first sentence should identify the artwork (name or type)\n"
    "and should be of the format: ARTWORK: [name]\n\n"

    "FORMAT CONSTRAINTS (VERY IMPORTANT):\n"
    "- Each sentence must end with a period followed by a space\n"
    "- Do NOT use dots inside names (for example write 'IM Pei' instead of 'I.M. Pei')\n"
    "- Do NOT include abbreviations with dots\n\n"

    "Rules:\n"
    "- Speak ONLY about the artwork\n"
    "- Do NOT describe the entire scene\n"
    "- Do NOT repeat the same idea\n"
)

USER_PROMPT = "Identify this artwork and present it as a museum guide."

# Sentinel value to signal the end of the stream
STREAM_DONE = None


def _encode_image(image_path: str, max_size: int = 512) -> str:
    """Resize and encode an image to base64 JPEG."""
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=60)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def stream_guide_sentences(image_path: str, sentence_queue: queue.Queue) -> None:
    """Stream the guide script sentence by sentence into a queue.

    Each complete sentence is put into the queue as soon as it's detected.
    A STREAM_DONE sentinel is put at the end.

    Args:
        image_path: Path to the image file.
        sentence_queue: Queue to put sentences into.
    """
    path = Path(image_path)
    if not path.exists():
        sentence_queue.put(STREAM_DONE)
        raise FileNotFoundError(f"Image not found: {image_path}")

    base64_image = _encode_image(image_path)
    client = OpenAI()

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=300,
            stream=True,
        )
    except (APIConnectionError, APIError) as e:
        sentence_queue.put(STREAM_DONE)
        raise RuntimeError(f"OpenAI API error: {e}")

    buffer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            buffer += delta.content
            # Split on sentence-ending punctuation
            while True:
                match = re.search(r'[.!?](?:\s|$)', buffer)
                if not match:
                    break
                end = match.end()
                sentence = buffer[:end].strip()
                buffer = buffer[end:]
                if sentence:
                    sentence_queue.put(sentence)

    # Flush any remaining text
    remaining = buffer.strip()
    if remaining:
        sentence_queue.put(remaining)

    sentence_queue.put(STREAM_DONE)



def stream_guide_sentences_from_bytes(image_bytes: bytes, sentence_queue: queue.Queue, client) -> None:

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=300,
            stream=True,
        )
    except (APIConnectionError, APIError) as e:
        sentence_queue.put(STREAM_DONE)
        raise RuntimeError(f"OpenAI API error: {e}")

    max_sentences = 5
    count = 0
    buffer = ""

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            buffer += delta.content

            while True:
                import re
                match = re.search(r'[.!?](?:\s|$)', buffer)
                if not match:
                    break

                end = match.end()
                sentence = buffer[:end].strip()
                buffer = buffer[end:]

                # ---------------------------
                #  HANDLE NONE
                # ---------------------------
                if sentence.strip() == "NONE":
                    print("Got NONE skipping TTS.")
                    return  # stop processing this frame entirely


                # ---------------------------
                #  PUSH TO QUEUE
                # ---------------------------
                try:
                    sentence_queue.put(sentence, block=False)
                    count += 1

                    if count >= max_sentences:
                        return

                except queue.Full:
                    print(" sentence dropped (queue full)")
                    return  # stop early to avoid blocking

    # ---------------------------
    #  FINAL BUFFER FLUSH
    # ---------------------------
    if count < max_sentences and buffer.strip():
        try:
            sentence_queue.put(buffer.strip(), block=False)
        except queue.Full:
            pass