"""Text-to-speech module using OpenAI TTS."""

from pathlib import Path

from openai import OpenAI, APIError, APIConnectionError


def generate_sentence_audio(text: str, output_path: str, client) -> str:
    """Convert a single sentence to an MP3 file via OpenAI TTS.

    Args:
        text: The sentence to synthesize.
        output_path: Output path for the MP3 file.

    Returns:
        The path of the generated MP3 file.

    Raises:
        ValueError: If the text is empty.
        RuntimeError: If the API call fails.
    """
    if not text or not text.strip():
        raise ValueError("The text is empty.")


    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )
    except APIConnectionError:
        raise RuntimeError(
            "Unable to reach the OpenAI TTS API. Check your connection."
        )
    except APIError as e:
        raise RuntimeError(f"OpenAI TTS API error: {e}")

    output = Path(output_path)
    response.stream_to_file(output)

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("The generated audio file is empty or missing.")

    return str(output)
