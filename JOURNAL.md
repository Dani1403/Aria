# Journal de bord — Audioguide Aria

Projet : audioguide hands-free pour musée via lunettes Meta Aria Gen 1.
Détection de fixation → reconnaissance d'œuvre → génération de script → lecture audio.

---

## Comment utiliser ce journal

- **Entrées les plus récentes en haut.**
- Une entrée par session de travail (pas une par jour forcé).
- Signer avec son prénom.
- 4 champs : **Fait** / **Bloqué** / **Prochain** / **Décisions**. Si vide → `—`.
- Commiter le journal séparément du code : `git commit -m "journal: AAAA-MM-JJ"`.
- Avant de commencer une session : lire la dernière entrée de l'autre (2 min).

---

## 2026-05-20 — Arthur — Streaming Aria branché + fluidification

Première session avec les vraies lunettes branchées. Trois avancées :

### 1. Streaming Aria fonctionnel (workaround)

Au départ on voulait lancer le streaming Aria directement depuis notre `main.py` (intégrer l'appel SDK dans le pipeline). Ça ne partait pas — le stream ne s'initialisait pas correctement quand on l'embarquait dans notre code.

**Workaround adopté** : on lance d'abord le **code officiel Project Aria** (script de streaming fourni par Meta) dans un terminal séparé, puis on démarre notre `main.py` qui se connecte au flux déjà actif. Ça marche de bout en bout. Pas idéal en termes de DX (deux commandes à lancer dans l'ordre), mais ça débloque tout le reste — on peut enfin tester le pipeline complet sur du vrai flux Aria.

> À reprendre plus tard : comprendre pourquoi le lancement embarqué échoue (problème d'init du client SDK ? thread principal ? conflit asyncio/threading ?) pour pouvoir tout unifier dans une seule commande.

### 2. Réduction de la taille de la queue

Le pipeline était laggy : trop de frames en attente faisaient s'accumuler du retard entre ce que voyait l'utilisateur et ce qu'il entendait. On a **réduit la taille de la queue** (probablement `frame_q` ou `sentence_q`, à confirmer dans le code) — résultat : pipeline plus fluide, moins de décalage perçu.

Logique sous-jacente : avec une queue plus petite, on jette les frames anciennes plus vite et on traite ce qui est **réellement devant l'utilisateur maintenant**, plutôt que de descendre une pile de frames périmées.

### 3. Arrêt audio quand l'œuvre disparaît du champ

Avant : si l'utilisateur quittait une œuvre en plein milieu du guide, l'audio continuait jusqu'au bout (toutes les phrases déjà mises en queue TTS étaient jouées). Comportement frustrant — le visiteur passe à autre chose mais entend encore le commentaire de l'œuvre précédente.

**Solution** : quand on détecte que l'image (l'œuvre) a disparu du champ, on **vide la queue TTS**. L'audio s'arrête, le pipeline est prêt à enchaîner sur la prochaine œuvre sans traîner de résidu.

C'est un premier morceau de la "logique de continuation/arrêt" prévue pour les semaines 2-3 — implémenté plus tôt que prévu parce que ça devenait gênant dès qu'on testait sur flux réel.

**Fait :** streaming Aria opérationnel via lancement séparé du SDK officiel ; queue réduite → pipeline plus fluide ; vidage de la queue TTS quand l'œuvre sort du champ.
**Bloqué :** lancement du streaming Aria depuis notre `main.py` ne fonctionne pas, on contourne en lançant le SDK Meta à côté.
**Prochain :**
- Écrire un **script bash** pour simplifier le lancement (enchaîner automatiquement : SDK Aria officiel → notre `main.py`) afin de masquer le workaround "deux commandes" derrière un seul `./run.sh`.
- **Reprise du guide** : si l'utilisateur quitte une œuvre puis y revient plus tard, reprendre la lecture **là où elle s'était arrêtée** (et non rejouer depuis le début, ni considérer l'œuvre comme nouvelle). Implique de stocker par œuvre l'index de la dernière phrase jouée, et au retour : retrouver l'œuvre dans `seen_artworks`, recharger les phrases restantes, repartir de là. À articuler avec le vidage de queue TTS implémenté aujourd'hui (le "stop" doit mémoriser où on s'est arrêté).
- Commencer le benchmark TTS prévu cette semaine.

**Décisions :** on assume le workaround "deux commandes" pour l'instant — débloquer le reste prime sur l'élégance ; queue plus petite = on privilégie la fraîcheur des frames sur le débit.

---

## 2026-05-11 — Arthur — État des lieux initial

Première entrée du journal. Récap de tout ce qui a été construit jusqu'ici pour qu'on parte sur une base commune.

### Architecture actuelle du pipeline

Le système fonctionne en **streaming multi-thread** avec 4 workers qui communiquent via des queues :

```
[VRS worker] → vrs_q → [Frame worker] → frame_q → [Vision worker] → sentence_q → [TTS worker] → audio_q → [Main thread: playback]
```

- **`stream.py`** : `simulate_stream()` découpe une vidéo en chunks de 3s et les pousse dans `vrs_q` en temps réel. Simule un flux Aria en attendant les vraies lunettes.
- **`extract_frames.py`** : `extract_frames_from_video()` échantillonne à 0.5 FPS (1 frame toutes les 2s) depuis les chunks. Version VRS aussi prête (`extract_frames()` via `pyvrs`).
- **`vision.py`** : `stream_guide_sentences_from_bytes()` envoie l'image à **GPT-4o-mini Vision** en mode streaming. Le prompt système force le format `ARTWORK: [nom]` + 3-5 phrases descriptives, ou `NONE` si rien de notable. Les phrases sont parsées au vol (regex sur `.!?`) et poussées une par une dans `sentence_q` avec un timestamp.
- **`tts.py`** : `generate_sentence_audio()` appelle **OpenAI TTS-1** (voix `nova`) phrase par phrase, retourne les bytes MP3.
- **`audio.py`** : lecture via **pygame**, supporte fichiers et bytes en mémoire.
- **`main.py`** : orchestre tout. Gère aussi la déduplication d'œuvres et la mesure de latence.

### Ce qui marche aujourd'hui

- Pipeline bout-en-bout fonctionnel : vidéo en entrée → audio en sortie en français.
- Streaming **sentence-by-sentence** : la 1ère phrase joue pendant que GPT continue de générer.
- **Phrase d'ouverture générique** pré-générée ("This is a nice artwork let me tell you more about it !") jouée dès qu'une nouvelle œuvre est détectée, pour masquer la latence du vrai script.
- **Mesure de latence UX vs REAL** : on log le temps entre détection de l'œuvre et premier son (UX = phrase générique) et premier vrai contenu (REAL = description GPT).
- Backpressure : si `sentence_q` dépasse 5 éléments, le vision worker attend (évite de saturer la queue TTS).
- Limite à 4 phrases max par œuvre côté TTS (`MAX_SENTENCES`).
- Compression d'image avant envoi (max 512px, JPEG qualité 60) pour réduire la latence Vision.

### Problèmes rencontrés et résolus

**1. Reconnaissance répétée de la même œuvre → guide audio relancé en boucle**

Au début, dès qu'on tombait sur plusieurs frames consécutives montrant la même œuvre, le pipeline relançait à chaque fois la séquence complète : nouvelle requête Vision, nouveau script, nouvelle phrase d'ouverture, nouveau TTS. Résultat : le visiteur entendait *"This is a nice artwork..."* en boucle dès qu'il restait quelques secondes devant un tableau, et les descriptions se superposaient ou se répétaient.

**Solution actuelle** (dans `main.py`) : un `set` `seen_artworks` accumule les œuvres déjà traitées (nom normalisé : lowercase, sans ponctuation). À chaque nouveau `ARTWORK:` détecté, on appelle `is_similar_artwork()` qui calcule un ratio de mots communs (seuil 50%). Si match → on saute. Exemple : "louvre pyramid" vs "glass pyramid" partagent "pyramid" → 50% → considéré comme la même œuvre.

> ⚠️ **À améliorer** : le `#TODO : NOT GOOD` dans le code. Le matching par mots est fragile (faux positifs sur des mots génériques, faux négatifs sur des reformulations). À terme on devrait passer à un matching visuel (embeddings CLIP ou features ORB sur l'image de référence captée au premier `ARTWORK:`) — ça résoudra aussi la question "l'œuvre est-elle toujours dans le champ ?" qu'on aura à traiter pour la logique de continuation.

**2. Passage du single-video au streaming multi-chunk**

Au départ, le code prenait **une seule vidéo en entrée** et la traitait en bloc : on extrayait toutes les frames d'un coup, on les envoyait à Vision, on attendait, on jouait l'audio. Workflow OK pour du test, mais incompatible avec un usage temps réel : impossible de réagir à un flux continu venant des lunettes.

**Refactor effectué** : introduction de `stream.py` + queue `vrs_q`. Maintenant le pipeline est **producteur/consommateur** :
- Un thread produit des chunks vidéo (`simulate_stream`) au fur et à mesure.
- Un thread consomme les chunks et en extrait les frames.
- Les frames partent en vision dès qu'elles sont prêtes.

Avantage : quand on branchera les Aria, il suffira de remplacer `simulate_stream` par un vrai pull depuis les lunettes (`pull_aria_recording` dans `utils.py` est déjà ébauché). Le reste du pipeline ne change pas.

### Outils et tech actuels

- **Vision** : `gpt-4o-mini` via l'API OpenAI Chat Completions, mode streaming, `detail: low`.
- **TTS** : `tts-1` (OpenAI), voix `nova`, sortie MP3.
- **CV** : OpenCV (lecture vidéo, encodage JPEG).
- **Audio** : pygame mixer.
- **VRS** : `pyvrs` (SDK Aria) pour quand les lunettes seront là.
- **Threading** : `threading` + `queue` standard Python.
- **Env** : `.env` (clé OpenAI via `python-dotenv`).

### Chantiers à venir (rappel du programme convenu)

- **Cette semaine** : réception et prise en main des Aria, benchmark TTS rapide (Piper, ElevenLabs Flash, gpt-4o-mini-tts), variantes de phrases d'ouverture.
- **Semaines 2-3** : logique de continuation/arrêt (le guide parle tant que l'œuvre est dans le champ), guide court vs long, intégration MVP ↔ flux Aria réel.
- **Semaine 4** : robustesse (catalogué vs libre, échecs de reconnaissance, réseau dégradé).
- **Semaines 5-6** : test terrain en musée, itération, démo POC.

**Fait :** récap initial du projet, mise en place du journal.
**Bloqué :** —
**Prochain :** réception des Aria (semaine prochaine), lancer en parallèle le benchmark TTS.
**Décisions :** journal en markdown à la racine du repo, format léger en 4 champs, entrées récentes en haut.

---
