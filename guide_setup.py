#!/usr/bin/env python3
"""Visitor questionnaire — run by run.sh before the experience starts.

Asks 4 questions in English (language, age, art knowledge, description
length), builds the GuideProfile — the single source of truth for
personalisation — and writes it to .guide_profile.json for main.py.
The profile is frozen for the whole session.
"""

import sys

from guide_profile import (
    DEFAULTS,
    PROFILE_FILE,
    GuideProfile,
    save_profile,
)


def ask_choice(question, options, default_key):
    """Numbered single-choice question. Empty answer -> default."""
    while True:
        print(f"\n{question}")
        for i, (key, label) in enumerate(options, 1):
            marker = " (default)" if key == default_key else ""
            print(f"  {i}. {label}{marker}")
        answer = input("> ").strip()
        if not answer:
            return default_key
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        for key, label in options:
            if answer.lower() in (key, label.lower()):
                return key
        print("Please answer with one of the numbers above.")


def ask_age(default):
    while True:
        print("\nHow old are you?")
        answer = input("> ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= 120:
            return int(answer)
        print("Please enter your age as a number.")


def main():
    if not sys.stdin.isatty():
        print("[SETUP] No interactive terminal detected — keeping the default profile.")
        profile = GuideProfile()
    else:
        print("=" * 46)
        print("  Aria Audio Guide — visitor profile")
        print("=" * 46)

        try:
            language = ask_choice(
                "Which language should the guide speak?",
                [("en", "English"), ("fr", "French"), ("es", "Spanish")],
                DEFAULTS["language"],
            )
            age = ask_age(DEFAULTS["age"])
            knowledge = ask_choice(
                "What is your level of art knowledge?",
                [("novice", "Novice"), ("intermediate", "Intermediate"), ("expert", "Expert")],
                DEFAULTS["knowledge"],
            )
            length = ask_choice(
                "How long should the descriptions be?",
                [("short", "Short"), ("medium", "Medium"), ("long", "Long")],
                DEFAULTS["length"],
            )
        except EOFError:
            print("\n[SETUP] Input closed — keeping the default profile.")
            profile = GuideProfile()
        else:
            profile = GuideProfile(language=language, age=age, knowledge=knowledge, length=length)

    save_profile(profile)
    min_s, max_s = profile.sentence_range
    print(
        f"\n[SETUP] Profile saved to {PROFILE_FILE}: {profile.language_name}, "
        f"{profile.age} years old, {profile.knowledge}, {profile.length} descriptions "
        f"({min_s}-{max_s} sentences). Enjoy your visit!"
    )


if __name__ == "__main__":
    main()
