"""Audio playback module using pygame."""

import time
from pathlib import Path

import pygame


def init_audio() -> None:
    """Initialize the pygame mixer."""
    pygame.mixer.init()


def play_audio_file(file_path: str) -> None:
    """Play an MP3 file and wait until it finishes.

    Args:
        file_path: Path to the MP3 file.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If playback fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    except pygame.error as e:
        raise RuntimeError(f"Audio playback error: {e}")


def quit_audio() -> None:
    """Shut down the pygame mixer."""
    pygame.mixer.quit()
