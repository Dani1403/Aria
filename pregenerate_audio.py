"""One-time batch job: pregenerate every framing-phrase clip (opening,
re-entry, ending) for every language x age-group combination, and cache them
to disk under AUDIO_CACHE_DIR.

Idempotent and incremental: existing files are skipped, so rerunning after
adding a language or editing a phrase only generates what's missing.

main.py loads these clips from disk at startup instead of calling the TTS API
live on every run (see JOURNAL 2026-07-07 for the profile system this caches).
"""

from dotenv import load_dotenv
from openai import OpenAI

from guide_profile import AGE_GROUP_SAMPLE_AGE, LANGUAGES, GuideProfile, cached_audio_path
from tts import generate_sentence_audio

POOLS = ("opening", "reentry", "ending")


def pregenerate_all(client) -> None:
    generated = 0
    skipped = 0
    failed = 0

    for language in LANGUAGES:
        for age_group, sample_age in AGE_GROUP_SAMPLE_AGE.items():
            profile = GuideProfile(language=language, age=sample_age)

            for pool in POOLS:
                for text in profile.phrases[pool]:
                    path = cached_audio_path(language, age_group, pool, text)
                    if path.exists():
                        skipped += 1
                        continue

                    try:
                        audio_bytes = generate_sentence_audio(
                            text, client, profile.tts_voice, profile.tts_speed, profile.tts_instructions
                        )
                    except Exception as e:
                        print(f"[PREGEN] FAILED {language}/{age_group}/{pool}: {text!r} ({e})")
                        failed += 1
                        continue

                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(audio_bytes)
                    generated += 1
                    print(f"[PREGEN] {language}/{age_group}/{pool}: {text!r}")

    print(f"\n[PREGEN] Done. generated={generated} skipped(cached)={skipped} failed={failed}")


if __name__ == "__main__":
    load_dotenv()
    pregenerate_all(OpenAI())
