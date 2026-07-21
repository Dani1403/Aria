"""Museum artwork databases — optional grounding for the vision model.

A museum database is one JSON file per museum in museums/ (the file stem is
the museum id stored in the profile):

    {
      "id": "petit-palais",
      "name": "Petit Palais, Paris",
      "artworks": [
        {
          "id": "courbet-chien-noir",        # stable slug, unique in the file
          "title": "...",                     # canonical title, used verbatim in ARTWORK:
          "artist": "...",                    # optional
          "year": "...",                      # optional
          "type": "painting",                 # optional
          "room": "Salle 12",                 # optional, helps disambiguation
          "visual": "...",                    # optional, 1-2 sentences: what the CAMERA
                                              # sees. Add it only when the title alone is
                                              # not descriptive enough to match on.
          "notes": ["...", "..."]             # optional curated facts; add them when the
                                              # model confabulates about an obscure work
        }
      ]
    }

Only id and title are required: a bare title+artist list (fast to author from
the museum's own inventory, light in the prompt) is a perfectly valid
database. The intended workflow is iterative — start bare, test in the
museum, then enrich with `visual`/`notes` only the entries the model still
mismatches or invents facts about.

The selected museum's catalog is compiled into the vision system prompt once
at session start (vision.build_system_prompt). The prompt stays static for the
whole session so OpenAI prefix caching absorbs the extra tokens.
"""

import json
from pathlib import Path

MUSEUMS_DIR = Path("museums")

# Everything else is optional enrichment; without a title an entry can neither
# be matched nor announced.
REQUIRED_ARTWORK_FIELDS = ("id", "title")


def list_museums(museums_dir: Path = MUSEUMS_DIR) -> dict:
    """Map museum id -> display name for every readable JSON file in museums/."""
    museums = {}
    if not museums_dir.is_dir():
        return museums
    for path in sorted(museums_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[MUSEUM] skipping {path.name}: {e}")
            continue
        name = data.get("name")
        if not name:
            print(f"[MUSEUM] skipping {path.name}: missing 'name' field")
            continue
        museums[path.stem] = name
    return museums


def load_museum(museum_id: str, museums_dir: Path = MUSEUMS_DIR):
    """Load one museum database, dropping unusable artwork entries.

    Returns None when the file is missing, unreadable or empty, so a stale
    profile degrades to free improvisation instead of blocking the pipeline.
    """
    path = museums_dir / f"{museum_id}.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"[MUSEUM] could not load {path} ({e}) — running without a museum database.")
        return None

    artworks = []
    for entry in data.get("artworks", []):
        if not isinstance(entry, dict):
            print(f"[MUSEUM] skipping non-object artwork entry in {path.name}")
            continue
        missing = [k for k in REQUIRED_ARTWORK_FIELDS if not entry.get(k)]
        if missing:
            label = entry.get("id") or entry.get("title") or "?"
            print(f"[MUSEUM] skipping artwork {label!r}: missing {', '.join(missing)}")
            continue
        artworks.append(entry)

    if not artworks:
        print(f"[MUSEUM] {path} has no usable artworks — running without a museum database.")
        return None

    data["artworks"] = artworks
    print(f"[MUSEUM] loaded {len(artworks)} artworks from {path} ({data.get('name', museum_id)})")
    return data
