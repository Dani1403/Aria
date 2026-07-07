"""Visitor profile — single source of truth for personalisation.

The GuideProfile is collected by guide_setup.py at launch (4 questions) and
frozen for the whole session. It drives three levers:
  - WHAT is said: vision.build_system_prompt() derives the output language,
    the age register (tone) and the knowledge depth from it,
  - HOW MUCH: the length preset caps the sentence count on both the vision
    prompt and the TTS side,
  - HOW it sounds: TTS voice/speed and the localized framing phrases below
    (opening, re-entry, ending, artwork name announcement).

Adding a language = adding an entry to LANGUAGES + a phrase set in PHRASES;
the vision prompt already handles any output language.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROFILE_FILE = ".guide_profile.json"

LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
}

KNOWLEDGE_LEVELS = ("novice", "intermediate", "expert")

# One representative age per group, used to pregenerate/look up cached audio.
# Voice/speed/instructions only depend on the age group, not the exact age, so
# any age in the group's range yields the same clip.
AGE_GROUP_SAMPLE_AGE = {
    "child": 8,
    "teen": 15,
    "adult": 30,
    "senior": 70,
}

# Framing-phrase clips (opening/reentry/ending) are pregenerated once by
# pregenerate_audio.py and committed to the repo, then loaded from disk at
# startup instead of hitting the TTS API on every run.
AUDIO_CACHE_DIR = Path("tts_cache")

# Length preset -> (min, max) sentences. The max is enforced on BOTH sides:
# in the vision prompt and as the TTS cap (otherwise extra sentences would be
# silently dropped on one side, see JOURNAL 2026-06-29).
LENGTH_PRESETS = {
    "short": (5, 8),
    "medium": (9, 13),
    "long": (14, 18),
}

DEFAULTS = {
    "language": "en",
    "age": 30,
    "knowledge": "intermediate",
    "length": "medium",
}

# Voice/speed/style by age group. Tone/rhythm only — never ties age to knowledge.
# Voices are restricted to the six supported by every OpenAI TTS model
# (tts-1 and gpt-4o-mini-tts alike). The style string feeds the `instructions`
# parameter, only honoured by gpt-4o-* TTS models; with tts-1 the adaptation
# falls back to voice+speed alone.
_TTS_BY_AGE_GROUP = {
    "child": (
        "fable", 0.95,
        "a warm, playful storyteller voice full of wonder, as if sharing "
        "secrets with a young child",
    ),
    "teen": (
        "nova", 1.0,
        "a lively, energetic voice with a punchy rhythm, like an enthusiastic "
        "young guide",
    ),
    "adult": (
        "nova", 1.0,
        "a natural, confident museum-guide voice, engaging and conversational",
    ),
    "senior": (
        "shimmer", 0.9,
        "a calm, clear, unhurried voice that articulates carefully, like a "
        "seasoned docent",
    ),
}

# Localized framing phrases. Pre-generated as TTS pools at startup; only the
# profile's language is used for a given session.
PHRASES = {
    "en": {
        "opening": [
            "Let me tell you about this piece.",
            "Here's something worth knowing about this work.",
            "This one has quite a story behind it.",
            "Take a closer look — there's more here than meets the eye.",
            "Allow me to shed some light on this artwork.",
            "There's a fascinating history behind what you're looking at.",
            "Let me walk you through what makes this piece special.",
            "You've picked a great one — let me tell you more.",
            "This artwork has a lot to say. Let me help you hear it.",
            "Step closer — I'll fill you in on this piece.",
        ],
        "reentry": [
            "You've seen before — let's continue.",
            "Welcome back to this piece. There's more to discover here.",
            "You're back ! — let me pick up where we left off.",
            "Good to have you back at this piece. Let's keep going.",
            "You've returned to this artwork. Let me continue.",
            "Back again — there's still more to explore this artwork.",
        ],
        "ending": [
            "That's everything for this artwork — take your time.",
            "And that's the story of this artwork.",
            "I'll let you take it all in from here.",
            "That's all I have for this artwork !",
            "Take a moment to look at this piece with fresh eyes.",
            "There's plenty more to see in this artwork — I'll be here when you're ready !",
            "That wraps up this artwork. Enjoy the rest of your visit.",
            "Feel free to linger at this artwork — or move on when you're ready.",
        ],
        "name": "This is {name}.",
    },
    "fr": {
        "opening": [
            "Laissez-moi vous parler de cette œuvre.",
            "Voici quelque chose qui mérite qu'on s'y attarde.",
            "Celle-ci a toute une histoire derrière elle.",
            "Regardez bien — il y a plus à voir qu'il n'y paraît.",
            "Permettez-moi de vous éclairer sur cette œuvre.",
            "Il y a une histoire fascinante derrière ce que vous regardez.",
            "Laissez-moi vous montrer ce qui rend cette pièce si spéciale.",
            "Vous avez l'œil — laissez-moi vous en dire plus.",
            "Cette œuvre a beaucoup à raconter. Laissez-moi vous la faire entendre.",
            "Approchez-vous — je vous raconte tout sur cette pièce.",
        ],
        "reentry": [
            "Vous revoilà devant cette œuvre — reprenons.",
            "Content de vous retrouver ici. Il reste des choses à découvrir.",
            "Vous êtes de retour ! Reprenons là où nous nous étions arrêtés.",
            "Ravi de vous revoir devant cette pièce. Continuons.",
            "Vous êtes revenu vers cette œuvre. Laissez-moi poursuivre.",
            "Encore là — il reste des choses à explorer sur cette œuvre.",
        ],
        "ending": [
            "C'est tout pour cette œuvre — prenez votre temps.",
            "Et voilà l'histoire de cette œuvre.",
            "Je vous laisse en profiter tranquillement.",
            "C'est tout ce que j'avais sur cette œuvre !",
            "Prenez un instant pour la regarder d'un œil neuf.",
            "Il y a encore beaucoup à voir dans cette œuvre — je suis là quand vous voulez !",
            "Voilà qui conclut cette œuvre. Bonne suite de visite.",
            "Restez encore un peu devant cette œuvre — ou avancez quand vous êtes prêt.",
        ],
        "name": "Voici {name}.",
    },
    "es": {
        "opening": [
            "Déjame hablarte de esta obra.",
            "Aquí hay algo que vale la pena conocer.",
            "Esta pieza tiene toda una historia detrás.",
            "Mira con atención — hay más de lo que parece a simple vista.",
            "Permíteme contarte más sobre esta obra.",
            "Hay una historia fascinante detrás de lo que estás viendo.",
            "Déjame mostrarte qué hace tan especial esta pieza.",
            "Has elegido bien — déjame contarte más.",
            "Esta obra tiene mucho que decir. Déjame ayudarte a escucharla.",
            "Acércate — te cuento todo sobre esta pieza.",
        ],
        "reentry": [
            "Ya has estado aquí antes — sigamos.",
            "Bienvenido de nuevo a esta obra. Aún queda mucho por descubrir.",
            "¡Has vuelto! Retomemos donde lo dejamos.",
            "Qué bien tenerte de vuelta ante esta pieza. Continuemos.",
            "Has regresado a esta obra. Déjame continuar.",
            "De vuelta otra vez — todavía queda más por explorar en esta obra.",
        ],
        "ending": [
            "Eso es todo sobre esta obra — tómate tu tiempo.",
            "Y esa es la historia de esta obra.",
            "Te dejo disfrutarla con calma.",
            "¡Eso es todo lo que tengo sobre esta obra!",
            "Tómate un momento para mirarla con otros ojos.",
            "Hay mucho más que ver en esta obra — aquí estaré cuando quieras.",
            "Con esto terminamos esta obra. Disfruta el resto de tu visita.",
            "Quédate un rato más ante esta obra — o sigue cuando estés listo.",
        ],
        "name": "Esta obra es {name}.",
    },
}


@dataclass(frozen=True)
class GuideProfile:
    language: str = DEFAULTS["language"]      # key of LANGUAGES
    age: int = DEFAULTS["age"]
    knowledge: str = DEFAULTS["knowledge"]    # one of KNOWLEDGE_LEVELS
    length: str = DEFAULTS["length"]          # key of LENGTH_PRESETS

    @property
    def language_name(self) -> str:
        return LANGUAGES[self.language]

    @property
    def sentence_range(self):
        return LENGTH_PRESETS[self.length]

    @property
    def max_sentences(self) -> int:
        return self.sentence_range[1]

    @property
    def age_group(self) -> str:
        if self.age <= 12:
            return "child"
        if self.age <= 17:
            return "teen"
        if self.age < 65:
            return "adult"
        return "senior"

    @property
    def tts_voice(self) -> str:
        return _TTS_BY_AGE_GROUP[self.age_group][0]

    @property
    def tts_speed(self) -> float:
        return _TTS_BY_AGE_GROUP[self.age_group][1]

    @property
    def tts_instructions(self) -> str:
        style = _TTS_BY_AGE_GROUP[self.age_group][2]
        return (
            f"Speak {self.language_name} like a native speaker, "
            f"with {style}."
        )

    @property
    def phrases(self) -> dict:
        return PHRASES[self.language]


def cached_audio_path(language: str, age_group: str, pool: str, text: str) -> Path:
    """Content-addressed path for a pregenerated framing-phrase clip.

    Keyed by the phrase text (not its position in the list) so editing one
    phrase only invalidates that one file instead of desyncing the whole
    cache against pregenerate_audio.py.
    """
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return AUDIO_CACHE_DIR / language / age_group / pool / f"{key}.mp3"


def save_profile(profile: GuideProfile, path: str = PROFILE_FILE) -> None:
    with open(path, "w") as f:
        json.dump(
            {
                "language": profile.language,
                "age": profile.age,
                "knowledge": profile.knowledge,
                "length": profile.length,
            },
            f,
            indent=2,
        )


def load_profile(path: str = PROFILE_FILE) -> GuideProfile:
    """Load the profile written by guide_setup.py.

    Missing file or invalid fields fall back to DEFAULTS (field by field) so
    the pipeline always starts, even without the questionnaire.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[PROFILE] {path} not found — using default profile (run guide_setup.py to personalise).")
        data = {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"[PROFILE] could not read {path} ({e}) — using default profile.")
        data = {}

    def pick(key, valid):
        value = data.get(key, DEFAULTS[key])
        if value not in valid:
            print(f"[PROFILE] invalid {key} {value!r} — falling back to {DEFAULTS[key]!r}")
            value = DEFAULTS[key]
        return value

    try:
        age = int(data.get("age", DEFAULTS["age"]))
    except (TypeError, ValueError):
        print(f"[PROFILE] invalid age {data.get('age')!r} — falling back to {DEFAULTS['age']}")
        age = DEFAULTS["age"]

    return GuideProfile(
        language=pick("language", LANGUAGES),
        age=age,
        knowledge=pick("knowledge", KNOWLEDGE_LEVELS),
        length=pick("length", LENGTH_PRESETS),
    )
