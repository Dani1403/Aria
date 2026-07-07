"""Text-to-speech module using OpenAI TTS."""

from openai import APIError, APIConnectionError

# Switching model is a one-line change. `instructions` (voice style steering)
# is only supported by gpt-4o-* TTS models; tts-1 rejects the parameter, so it
# is only sent when the model supports it.
TTS_MODEL = "gpt-4o-mini-tts"


def generate_sentence_audio(text: str, client, voice: str = "nova", speed: float = 1.0, instructions: str = None):
    """Convert a single sentence to an MP3 file via OpenAI TTS.

    Args:
        text: The sentence to synthesize.
        client: The client model
        voice: OpenAI TTS voice (driven by the visitor profile).
        speed: Speech rate multiplier (driven by the visitor profile).
        instructions: Voice style directions (driven by the visitor profile).
            Only sent when TTS_MODEL supports them (gpt-4o-* models).

    Returns:
        The audio bytes

    Raises:
        ValueError: If the text is empty.
        RuntimeError: If the API call fails.
    """
    if not text or not text.strip():
        raise ValueError("The text is empty.")

    kwargs = dict(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        speed=speed,
    )
    if instructions and TTS_MODEL.startswith("gpt-"):
        kwargs["instructions"] = instructions

    try:
        response = client.audio.speech.create(**kwargs)
    except APIConnectionError:
        raise RuntimeError(
            "Unable to reach the OpenAI TTS API. Check your connection."
        )
    except APIError as e:
        raise RuntimeError(f"OpenAI TTS API error: {e}")


    audio_bytes = response.content

    return audio_bytes
