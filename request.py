import base64
import time
from pathlib import Path
from io import BytesIO

from PIL import Image
from openai import OpenAI

client = OpenAI()


def compress_image(image_bytes, max_size=512, jpeg_quality=70):
    """
    Resize and compress image to reduce API latency.
    """

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.rotate(-90, expand=True)

    # resize while keeping aspect ratio
    img.thumbnail((max_size, max_size))

    buffer = BytesIO()

    img.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality)

    compressed_bytes = buffer.getvalue()

    print(f"Original size: {len(image_bytes)/1024:.1f} KB")
    print(f"Compressed size: {len(compressed_bytes)/1024:.1f} KB")

    return compressed_bytes


def analyze_image(image_bytes):

    start = time.perf_counter()

    print("Compressing image...")

    image_bytes = compress_image(image_bytes)

    print("Encoding image...")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    print("Sending request to OpenAI vision model...")

    vision_start = time.perf_counter()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Identify and create a brief description of the scene i one phrase. also say hi to Daniel he is the big boss."
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    },
                ],
            }
        ],
        max_output_tokens=300,
    )

    vision_time = time.perf_counter() - vision_start

    text_output = response.output_text

    total_time = time.perf_counter() - start

    print(f"\nVision model latency: {vision_time:.3f} seconds")
    print(f"Total analysis time: {total_time:.3f} seconds\n")

    return text_output


def text_to_speech(text):

    print("Generating speech...")

    tts_start = time.perf_counter()

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )

    tts_time = time.perf_counter() - tts_start

    output_path = Path("audio.mp3")

    with open(output_path, "wb") as f:
        f.write(speech.content)

    print(f"Audio saved to: {output_path}")
    print(f"TTS latency: {tts_time:.3f} seconds\n")


if __name__ == "__main__":

    pipeline_start = time.perf_counter()

    print("\n--- TEST START ---\n")

    with open("mona.jpg", "rb") as f:
        image_bytes = f.read()

    explanation = analyze_image(image_bytes)

    print("\nMODEL RESPONSE:\n")
    print(explanation)

    text_to_speech(explanation)

    pipeline_total = time.perf_counter() - pipeline_start

    print(f"\nFull pipeline time: {pipeline_total:.3f} seconds")

    print("\n--- DONE ---\n")