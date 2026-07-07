"""Artwork recognition module using GPT-4o Vision with streaming."""

import base64
import io
import queue
import re
from pathlib import Path

from PIL import Image
from openai import OpenAI, APIError, APIConnectionError

from guide_profile import GuideProfile

# Register by age group: tone, vocabulary, sentence length, rhythm ONLY.
# Knowledge depth is a separate, independent axis (a child can be an expert,
# a senior a novice) — never mix the two here.
AGE_REGISTERS = {
    "child": (
        "Your listener is a young child: use a warm, playful, enthusiastic tone, "
        "simple everyday words, short sentences, and vivid comparisons with familiar things."
    ),
    "teen": (
        "Your listener is a teenager: use a lively, direct, engaging tone with a "
        "punchy rhythm, and highlight surprising or dramatic details."
    ),
    "adult": (
        "Your listener is an adult: use a natural, engaging, conversational tone "
        "with a steady rhythm."
    ),
    "senior": (
        "Your listener is an older adult: use a calm, unhurried, clearly structured "
        "delivery with a measured rhythm, and avoid slang."
    ),
}

# Depth by knowledge level: terminology and assumed background ONLY.
KNOWLEDGE_DEPTHS = {
    "novice": (
        "They are new to art: assume no prior art knowledge, avoid technical terms "
        "or explain them at once in plain words, and focus on the story and what to look at."
    ),
    "intermediate": (
        "They have an intermediate knowledge of art: assume familiarity with major "
        "periods and styles, use common art terms without defining them, and connect "
        "the work to its movement and context."
    ),
    "expert": (
        "They are an art expert: use precise terminology freely, skip the basics, "
        "and go deeper into technique, attribution, provenance and comparisons with "
        "related works."
    ),
}


def build_system_prompt(profile: GuideProfile) -> str:
    """Build the vision system prompt from the visitor profile.

    The profile drives the output language, the age register (tone), the
    knowledge depth (terminology, assumed background) and the sentence count
    (length preset). Control tokens (ARTWORK:, NONE, END) stay in English
    whatever the output language, so the pipeline parsing never changes.
    """
    min_s, max_s = profile.sentence_range
    return (
        "You are a tour guide in an augmented reality system.\n\n"

        "Your task:\n"
        "- If you see a genuine work of art or heritage sight (painting, sculpture, "
        "installation, monument, landmark, famous building, historic site), describe "
        "it as a guide.\n"
        "- It may be small, partially visible, behind glass, or in a crowded scene.\n\n"

        "IMPORTANT:\n"
        "- A genuine artwork does NOT need to be perfectly identified: if it is "
        "clearly an artwork but you are unsure which one, still describe it\n"
        "- But never treat an ordinary object as an artwork just to have something "
        "to say\n\n"

        "CRITICAL RULE:\n"
        "- Ordinary room features and equipment are NEVER notable: plain ceilings, "
        "lights and lamps, computers, cables, furniture, doors, windows, "
        "floors, walls, signs, barriers, people. Do not describe them.\n"
        "- A REPRODUCTION of an artwork counts as the artwork itself: if a "
        "painting or sculpture is shown on a screen, tablet, poster, postcard or "
        "book page, describe the ARTWORK (never the device or medium showing it). "
        "A screen showing anything else (code, text, apps, video) is NOT notable.\n"
        "- Decorated architecture only counts if it is itself a work of art "
        "(for example a painted fresco ceiling in a palace), not an ordinary "
        "decorated surface.\n"
        "- If there is no artwork or notable sight, or you doubt the thing is an "
        "artwork at all, output exactly: NONE\n"
        "- Output ONLY the word NONE (no punctuation, no explanation)\n\n"

        "If there is something notable:\n"
        f"- Provide a rich, detailed explanation ({min_s} to {max_s} sentences)\n"
        "- Cover what it is, who made it, when, the style or technique, its history, "
        "and an interesting anecdote or detail\n"
        "- The first sentence should identify it (name or type)\n"
        "and should be of the format: ARTWORK: [name].\n\n"

        "AUDIENCE (VERY IMPORTANT):\n"
        f"- Write every description sentence in {profile.language_name}.\n"
        f"- {AGE_REGISTERS[profile.age_group]}\n"
        f"- {KNOWLEDGE_DEPTHS[profile.knowledge]}\n"
        "- Age and knowledge are independent: never infer how much the listener knows "
        "about art from their age. Age sets the tone, knowledge sets the depth.\n\n"

        "FORMAT CONSTRAINTS (VERY IMPORTANT):\n"
        "- Each sentence must end with a period followed by a space\n"
        "- Do NOT use dots inside names (for example write 'IM Pei' instead of 'I.M. Pei')\n"
        "- Do NOT include abbreviations with dots\n"
        "- ARTWORK:, NONE and END are control tokens: always write them exactly as "
        "shown, in English, whatever the output language\n"
        "- After your very last sentence, output exactly: END\n"
        "- Output ONLY the word END (no punctuation, no explanation)\n\n"

        "Rules:\n"
        "- Speak ONLY about the notable thing\n"
        "- Do NOT describe the entire scene\n"
        "- Do NOT repeat the same idea\n"
    )


# Prompt for the default profile — fallback when no profile is provided
# (legacy path, tests).
SYSTEM_PROMPT = build_system_prompt(GuideProfile())

USER_PROMPT = (
    "If this image shows an artwork or notable sight, identify it and present it "
    "as a museum guide. If not, output NONE."
)

# Sentinel value to signal the end of the stream
STREAM_DONE = None


def _encode_image(image_path: str, max_size: int = 512) -> str:
    """Resize and encode an image to base64 JPEG."""
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=60)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")



def stream_guide_sentences_from_bytes(image_bytes: bytes, timestamp: float, sentence_queue: queue.Queue, client, max_sentences: int=10, system_prompt: str=None) -> None:

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    try:
        stream = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
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
            max_completion_tokens=800,
            stream=True,
        )
    except (APIConnectionError, APIError) as e:
        print(f"Vision API error on frame, skipping: {e}")
        return

    count = 0
    buffer = ""

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            buffer += delta.content

            while True:
                match = re.search(r'[.!?](?:\s|$)', buffer)
                if not match:
                    break

                end = match.end()
                sentence = buffer[:end].strip()
                buffer = buffer[end:]

                # ---------------------------
                #  HANDLE NONE / END
                # ---------------------------
                if sentence.strip() == "NONE":
                    print("Got NONE skipping TTS.")
                    return  # stop processing this frame entirely

                if sentence.strip().rstrip('.!?') == "END":
                    try:
                        sentence_queue.put(("END", timestamp), block=False)
                    except queue.Full:
                        pass
                    return

                # ---------------------------
                #  PUSH TO QUEUE
                # ---------------------------
                try:
                    sentence_queue.put((sentence, timestamp), block=False)
                    count += 1

                    if count >= max_sentences:
                        return

                except queue.Full:
                    print(" sentence dropped (queue full)")
                    return

    # ---------------------------
    #  FINAL BUFFER FLUSH
    # ---------------------------
    remaining = buffer.strip()
    if remaining and count < max_sentences:
        tokens = [t.rstrip('.!?') for t in re.split(r'[\s\n]+', remaining) if t.strip()]
        if set(tokens) <= {"NONE", "END"}:
            if "END" in tokens:
                try:
                    sentence_queue.put(("END", timestamp), block=False)
                except queue.Full:
                    pass
        else:
            try:
                sentence_queue.put((remaining, timestamp), block=False)
            except queue.Full:
                pass

#Legacy function
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
            model="gpt-5.4-mini",
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
            max_completion_tokens=800,
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


