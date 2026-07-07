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
## 2026-07-07 — Arthur — Refonte marche/pause/reprise + prompt vision + voix par profil

### 1. Logique marche/pause/reprise — spec clarifiée et réimplémentée

Les vieux problèmes étaient revenus (l'audio ne s'arrêtait pas toujours en marchant, pas d'enchaînement sur l'œuvre suivante à l'arrêt). Spec retenue : je marche → le guide finit sa phrase et s'arrête ; je m'arrête → nouvelle œuvre = il la présente, œuvre déjà vue = il rejoue la suite non entendue (**jamais de regénération** — si le guide est fini, silence) ; la bascule d'œuvre **sans marcher** n'est autorisée que si le guide de l'œuvre courante est terminé. Enchaînements libres (vue → nouvelle → retour sur vue).

Correctifs structurels dans `main.py`/`motion.py` :

- **La pause est posée par le détecteur IMU lui-même** (`WalkingDetector(pause_event=paused)`) à l'instant du départ de marche. Avant, c'était le `tts_worker` qui la posait — or il passe des secondes bloqué dans chaque appel TTS (gpt-4o-mini-tts est plus lent que tts-1, ce qui a fait resurgir le bug) et la lecture continuait pendant ce temps.
- **Compteur de départs de marche** (`walk_events`) consommé par le `tts_worker` au lieu d'une comparaison d'états : une marche courte ne passe plus inaperçue, la sauvegarde des phrases restantes se fait toujours.
- **Filtre de fraîcheur** (`last_still_time`) : toute phrase issue d'une frame capturée avant le dernier passage marche→immobile est ignorée (headers compris) — fini les streams périmés qui polluaient l'état. Le `vision_worker` **saute carrément l'analyse** des frames périmées : c'est le gros gain de réactivité pour l'enchaînement (une analyse complète dure 10-20 s).
- La reprise ne regénère plus jamais (`allow_description=False` en re-entrée, l'œuvre passe en `description_complete`), une phrase dont la synthèse se termine pendant un départ de marche part en réserve au lieu de jouer, la lecture re-vérifie la pause toutes les 0.1 s même en attente de clip, et `paused.clear()` remonte avant la génération du clip NAME (sinon +1 appel TTS de latence UX à chaque œuvre).
- **Déduplication durcie** : mots vides EN/FR/ES retirés du matching et seuil 0.5 → 0.6. « Portrait de Louis XIV » vs « Portrait de Napoléon » ne fusionnent plus (avant : la 2e œuvre restait muette). Reste par nom — le vrai fix (matching visuel) est toujours en TODO.

### 2. Seuil IMU recalibré

`enter_threshold` = **0.75** (remplace le 0.6 calé le 2026-06-29 — c'est la valeur validée sur flux réel aujourd'hui), `exit=0.3`, dwell 0.4 s inchangés.

### 3. Prompt vision — faux positifs et reproductions

Le guide décrivait plafonds, luminaires, ordinateurs. Causes : le prompt poussait au rappel maximal (« If it looks notable, assume it is », « Prefer describing rather than missing ») et le user prompt présupposait une œuvre (« Identify this artwork… ») — gpt-5.4-mini suivait docilement. Correctifs dans `build_system_prompt` :

- Liste d'exclusion explicite (plafonds ordinaires, lampes, ordinateurs, mobilier, portes, personnes…), doute **asymétrique** : doute que ce soit une œuvre → NONE ; œuvre certaine mais non identifiée → décrire quand même. Exception préservée pour l'architecture-œuvre (plafond à fresques).
- **Piège découvert en test** : la première version excluait les « écrans » → le protocole de test (œuvres affichées plein écran sur iPad) renvoyait NONE sur Le Cri. Règle ajoutée : une **reproduction** (écran, poster, carte postale, page de livre) compte comme l'œuvre elle-même — on décrit l'œuvre, jamais l'appareil ; un écran montrant autre chose (code, apps) reste non-notable. Validé : reconnaissance OK sur iPad.

### 4. Voix adaptée au profil visiteur

- Voix distincte par groupe d'âge (`fable` enfant, `nova` ado/adulte, `shimmer` senior — avant tout le monde sauf enfant avait `nova`).
- Nouvelle propriété `GuideProfile.tts_instructions` : accent natif dans la langue du profil + style vocal par âge (conteur chaleureux / rythme punchy / guide posé / débit calme articulé), passée partout via le helper `tts()`. Le paramètre `instructions` n'est envoyé que si `TTS_MODEL` le supporte (gpt-4o-*) ; retour à tts-1 = une ligne, rien ne casse. NB : `speed` est ignoré par gpt-4o-mini-tts, le rythme passe par les instructions.

**Fait :** refonte complète marche/pause/reprise (validée sur flux réel : arrêt en fin de phrase, enchaînement, reprise, silence si guide fini) ; seuil IMU 0.75 ; prompt vision anti-faux-positifs + règle reproductions ; voix/style TTS par profil.
**Bloqué :** —
**Prochain :**
- Matching visuel des œuvres (le matching par nom traite « Mona Lisa » vs « La Joconde » comme deux œuvres).
- TTS en streaming pour absorber la latence de gpt-4o-mini-tts.
- Valider les voix/styles par langue à l'oreille (accent FR/ES).
**Décisions :** reprise = **rejouer l'existant uniquement**, jamais de regénération ; bascule sans marche seulement si guide courant terminé ; les reproductions d'œuvres (écran/poster) comptent comme l'œuvre ; IMU enter=0.75 est la valeur de référence.

---
## 2026-07-07 — Arthur — Benchmark modèles vision/TTS (gpt-5.4-mini, gpt-4o-mini-tts)

### 1. Migration des modèles

- **Vision** : `gpt-4o-mini` → **`gpt-5.4-mini`** dans `vision.py` (les deux appels). Piège de migration : la famille gpt-5 refuse `max_tokens` (erreur 400 `Unsupported parameter`) → remplacé par `max_completion_tokens=800`. Attention : les tokens de raisonnement interne comptent dans ce budget ; si des descriptions sortent tronquées, ajouter `reasoning_effort="minimal"` ou augmenter le plafond.
- **TTS** : testé **`gpt-4o-mini-tts`** en remplacement de `tts-1` (même signature d'appel : `voice`/`speed` acceptés ; offre en plus un paramètre `instructions` pour piloter le ton — intéressant pour le registre enfant).

### 2. Benchmark latence — 3 configurations

Conditions : configuration de base (profil par défaut), lunettes connectées en **WiFi**, mêmes conditions exactes pour les trois runs, une mesure par config. UX = délai détection d'œuvre → premier son (clip générique) ; REAL = délai → première vraie phrase de description.

| Vision | TTS | UX | REAL |
|---|---|---|---|
| **gpt-5.4-mini** | **tts-1** | **2.98 s** | **7.75 s** |
| gpt-4o-mini | tts-1 | 4.58 s | 10.0 s |
| gpt-5.4-mini | gpt-4o-mini-tts | 4.12 s | 9.29 s |

Lecture :

- **gpt-5.4-mini en vision = gain net** : −1.6 s UX / −2.2 s REAL à TTS égal. Le premier token (header `ARTWORK:`) sort nettement plus vite.
- **gpt-4o-mini-tts coûte ~1.2–1.5 s** vs tts-1 : attendu sur REAL (l'annonce du nom et les phrases sont synthétisées à la volée, TTFB plus élevé). L'écart UX est en partie suspect : le clip générique est pré-généré, donc le TTS ne devrait pas jouer — soit variance entre runs, soit attente sur le pool de pré-génération (plus lent avec gpt-4o-mini-tts) si l'œuvre est détectée très tôt.
- Récupérable si on veut la voix gpt-4o-mini-tts : passer la lecture en **streaming** (`client.audio.speech.with_streaming_response` + lecture par chunks) au lieu d'attendre le MP3 complet — devrait regagner plus que le surcoût.

### 3. Incidents de session

- `tts.py`/`vision.py` écrasés par une version antérieure (ré-extraction de l'archive) → `generate_sentence_audio()` avait reperdu ses paramètres `voice`/`speed` (crash `takes 2 positional arguments but 4 were given` à la pré-génération, car `main.py` les passe depuis le profil). Restauré. Le repo local n'est pas un dépôt git → tout écrasement est silencieux.
- Streaming WiFi sur le réseau campus (132.69.x.x) inutilisable : spam `CRITICAL DDS: sample lost` sur le topic RGB, ~2 frames délivrées en 35 s, le pipeline tourne mais la vision n'a rien à décrire. Le hotspot iPhone (172.20.10.x) reste le setup de référence.

**Fait :** vision migrée sur gpt-5.4-mini (+ fix `max_completion_tokens`) ; benchmark latence 3 configs (tableau ci-dessus) ; restauration des fichiers écrasés ; diagnostic du WiFi campus.
**Bloqué :** streaming WiFi sur réseau institutionnel (perte massive de samples DDS) — contourné via hotspot.
**Prochain :**
- Refaire les mesures sur 3–5 runs par config (une seule mesure = variance API non maîtrisée, surtout l'écart UX de la config 3).
- Si la voix gpt-4o-mini-tts est retenue pour le profil enfant : implémenter le TTS en streaming pour absorber le surcoût de latence.
- Surveiller les descriptions tronquées avec gpt-5.4-mini (reasoning tokens dans le budget de 800) ; le cas échéant `reasoning_effort="minimal"`.
**Décisions :** config retenue = **vision gpt-5.4-mini + TTS tts-1** (meilleure latence : UX 2.98 s / REAL 7.75 s) ; gpt-4o-mini-tts écarté pour l'instant, à réévaluer avec le streaming audio.

---
## 2026-07-07 — Arthur — Personnalisation du guide (profil visiteur)

### 1. Questionnaire au lancement (`guide_setup.py`)

`run.sh` pose maintenant 4 questions (en anglais) avant de démarrer l'expérience : langue (English/French/Spanish), âge, niveau de connaissance en art (Novice/Intermediate/Expert), longueur des descriptions (Short/Medium/Long). Les réponses sont écrites dans `.guide_profile.json`, que `main.py` charge via `--profile-file`. Entrée vide = valeur par défaut ; pas de terminal interactif (stdin non-tty) = profil par défaut, le pipeline démarre quand même.

### 2. `GuideProfile` — source de vérité unique (`guide_profile.py`)

Nouveau module : dataclass gelée `GuideProfile` (language/age/knowledge/length) + propriétés dérivées. Le profil est figé pour toute la session et se propage à trois leviers :

- **Ce qui est dit** : `vision.build_system_prompt(profile)` construit le prompt système dynamiquement — langue de sortie imposée, registre lié à l'âge (child ≤12 / teen / adult / senior 65+ : ton, vocabulaire, rythme) et profondeur liée au niveau de connaissance (terminologie, savoir présumé). Les deux axes sont **indépendants** : consigne explicite dans le prompt pour que l'âge ne détermine jamais la profondeur (un enfant peut être expert). Les tokens de contrôle `ARTWORK:`/`NONE`/`END` restent en anglais quelle que soit la langue — le parsing du pipeline ne change pas.
- **Combien** : presets de longueur Short 5-8 / Medium 9-13 / Long 14-18 phrases, injectés dans le prompt vision **et** utilisés comme plafond côté TTS (`profile.max_sentences` remplace le global `MAX_SENTENCES`) — les deux plafonds restent alignés.
- **Comment ça sonne** : voix et débit TTS dérivés du groupe d'âge (child → `fable` à 0.95, senior → `nova` à 0.9, sinon `nova` à 1.0), et les trois pools de phrases d'habillage (ouverture, re-entry, fin) + l'annonce du nom (« This is X. » / « Voici X. » / « Esta obra es X. ») sont localisés EN/FR/ES dans `PHRASES`.

Ajouter une langue = ajouter une entrée `LANGUAGES` + un jeu de phrases dans `PHRASES` ; le prompt vision gère déjà n'importe quelle langue.

### 3. Refactors associés

`tts.generate_sentence_audio()` prend `voice`/`speed` en paramètres (défauts inchangés). `vision.stream_guide_sentences_from_bytes()` prend un `system_prompt` optionnel (fallback sur le prompt du profil par défaut). `main.py` parse `--profile-file`, loggue le profil au démarrage (`[PROFILE] ...`) et le `tts_worker` passe par un helper `tts()` qui applique voix/débit partout (pools pré-générés inclus). `load_profile()` est robuste : fichier absent ou champ invalide → fallback champ par champ sur les défauts avec warning.

Au passage : fix d'un `\n` manquant dans le prompt d'origine (les lignes « output exactly: END » et « Output ONLY the word END » étaient concaténées).

**Fait :** questionnaire au lancement ; `GuideProfile` propagé aux trois leviers (prompt vision, plafond de phrases, voix/débit + phrases localisées EN/FR/ES) ; testé : parcours interactif (pty), non-interactif, profils corrompus/absents, prompt généré vérifié.
**Bloqué :** —
**Prochain :**
- Tester sur flux réel que GPT-4o-mini respecte bien la langue et le registre (surtout child + expert, combinaison inhabituelle).
- Valider les voix TTS par langue (nova/fable sur du français et de l'espagnol — accent correct ?).
- Éventuellement : centres d'intérêt du visiteur comme 5e question (prévu dans l'entrée du 2026-06-29).
**Décisions :** le profil est **figé pour la session** (choix au lancement, pas de changement en cours de visite) ; l'âge et le niveau de connaissance sont **deux axes indépendants** dans le prompt ; les tokens de contrôle restent en anglais pour ne pas toucher au parsing.

---
## 2026-06-30 — Daniel — Réorganisation du repo + phrases pré-générées + signal END + debug

### 1. Nettoyage et réorganisation du repo

Gros rangement : suppression des vidéos de test lourdes (`Louvre.mp4`, `Louvre2.mp4`, `Stream.mp4`), regroupement des assets statiques dans `assets/` (images de t
est, fichiers audio, `Stream.vrs.json`), et archivage des scripts désormais inutiles dans `legacy/` (`extract_frames.py`, `request.py`, `stream.py`, `utils.py`, `
vrs_to_video.py`). `common.py` et `visualizer.py` supprimés. Le repo est maintenant propre et ne contient que ce qui est actif.

Refactor associé dans `audio.py`, `main.py`, `tts.py`, `vision.py` pour supprimer les imports et références aux fichiers disparus.

### 2. Pool de phrases pré-générées (ouverture + re-entry + fin)

Ajout d'un troisième pool de phrases : `ENDING_SENTENCES` — phrases de conclusion jouées quand le modèle vision a fini de décrire une œuvre (« That's everything f
or this one », « I'll let you take it all in from here »…). Pré-générées en TTS au démarrage comme les deux autres pools, stockées dans `ending_audio_pool`.

Correction de la structure des pools : les entrées passent de `audio_bytes` à `(text, audio_bytes)` pour permettre le logging — on sait maintenant quelle phrase e
xacte a été tirée au sort à chaque lecture.

### 3. Signal END — le modèle annonce qu'il a fini

Dans `vision.py` : ajout d'une consigne dans le system prompt demandant au modèle de terminer sa description par le mot exactement `END` (sans ponctuation). Ce si
gnal est parsé au fil du stream et poussé dans la queue comme `("END", timestamp)`.

Dans `main.py` : quand le `tts_worker` reçoit `END`, il pioche une phrase de conclusion dans `ending_audio_pool` et la pousse dans `audio_q` avec le type `"END"`.

Gestion du flush final dans `vision.py` : si `END` tombe dans le dernier buffer résiduel, il est correctement renvoyé sans être traité comme une vraie phrase.

### 4. Logging TTS amélioré

Toutes les traces `print` du `tts_worker` sont maintenant préfixées `[TTS]` avec le type (`GENERIC`, `REENTRY`, `NAME`, `REAL`, `END`) et le texte prononcé. Plus
facile de suivre ce qui est joué dans les logs.

### 5. Fix run.sh — caractères `\r` (Windows → Linux)

`run.sh` avait des fins de ligne Windows (`\r\n`) qui causaient des erreurs silencieuses à l'exécution sous Linux/macOS. Nettoyé.

**Fait :** repo nettoyé (legacy/, assets/, vidéos supprimées) ; pool de phrases de fin pré-générées ; signal END du modèle vision → phrase de conclusion automatiq
ue ; logging [TTS] avec texte ; fix \r dans run.sh.
**Bloqué :** —
**Prochain :**
- Tester le signal END sur flux réel (vérifier que le modèle envoie bien END de façon fiable).
- Éventuellement : paralléliser l'analyse des frames (plusieurs threads vision) pour réduire la latence de la première description.
**Décisions :** le signal END remplace un timer ou un compteur de phrases pour savoir quand le modèle a fini — plus propre et aligné avec la génération réelle ; l
es pools stockent désormais `(text, bytes)` pour que les logs soient utiles.

---
---
## 2026-06-29 — Arthur — Déclencheur IMU (marche) + annonce du nom d'œuvre + descriptions longues

### 1. Mise en place de l'IMU — la marche pilote l'audio

Nouveau module **`motion.py`** (`WalkingDetector`) : on s'abonne au flux **IMU** des lunettes (`StreamingDataType.Imu`, activé dans `main.py` — il était commenté) et on détecte la marche via l'**écart-type de la norme de l'accéléromètre** sur une fenêtre glissante d'1 s. Immobile → std proche de 0 (gravité seule) ; marche → oscillations périodiques. Hystérésis (deux seuils `enter`/`exit`) + temps minimal dans un état pour éviter le clignotement. Le détecteur pilote un `threading.Event` `walking` partagé.

Script de debug **`imu_debug.py`** : affiche en direct `|a|`, le `std` glissant et le nombre d'échantillons par IMU. Sert à vérifier que le flux IMU arrive et à **régler les seuils** (lire le std en immobile / marche lente / marche normale). Seuils actuels après calage : `enter=0.6`, `exit=0.3` (abaissés pour capter aussi la **marche lente**).

> ⚠️ Le profil de streaming doit diffuser l'IMU. Si `imu_debug.py` n'affiche aucune donnée → changer de profil (ex. `./run.sh wifi profile12`).

### 2. Nouvelle logique audio : marche = pause, détection d'œuvre = reprise

Le déclencheur d'arrêt **« l'œuvre sort du champ » (`NONE`) est remplacé par le mouvement**. Comportement implémenté dans `tts_worker` + boucle de lecture, via deux signaux : `walking` (ignorer la vision pendant la marche) et un nouveau `paused` (bloque la lecture, levé **uniquement** quand le guide (re)lance une œuvre — pas à l'arrêt).

- **Plus de coupure sèche** : la phrase en cours **va jusqu'au bout** (retrait du `should_stop` ; le gate de pause est testé *avant* de tirer la phrase suivante).
- **Marche détectée** → on finit la phrase, on **sauvegarde les phrases restantes** dans `seen_artworks[œuvre]`, puis silence.
- **S'arrêter de marcher ne relance rien.** La reprise est déclenchée par la **détection d'une œuvre** une fois immobile : même œuvre → reprise **là où on s'était arrêté** (phrase de retour « welcome back » + phrases sauvegardées) ; œuvre différente → description de la nouvelle.
- `NONE` (hors champ) est devenu **inerte** : seul le mouvement met en pause.

### 3. Annonce du nom de l'œuvre (bug)

Le header `ARTWORK: [nom]` renvoyé par la vision était **consommé uniquement pour l'état** : seul un clip générique était joué, **le vrai nom n'était jamais prononcé** (constaté sur *La Cène* de Léonard de Vinci, mais le problème était général). Correctif : à la détection d'une **nouvelle œuvre**, on génère et joue « *This is {nom}.* » juste après le clip générique (qui masque la latence de génération), avec `raw_name` pour préserver la casse.

### 4. Descriptions plus longues

Relevé des trois plafonds qui se cumulaient : prompt vision « 5 to 10 » → **« 12 to 18 » phrases** (+ consigne de couvrir auteur, époque, technique, histoire, anecdote), `max_tokens` 300 → **800**, et `MAX_SENTENCES` 10 → **18** (sinon les phrases au-delà étaient jetées silencieusement côté vision **et** TTS).

**Fait :** IMU branché + détection de marche (`motion.py`, `imu_debug.py`) ; bascule du déclencheur audio « hors champ » → « mouvement » ; phrase en cours jamais coupée ; reprise sur détection d'œuvre (pas à l'arrêt) avec reprise au point d'arrêt ; annonce du nom d'œuvre corrigée ; descriptions allongées.
**Bloqué :** —
**Prochain (dans l'ordre) :**
- **Paralléliser l'analyse des frames** : plusieurs threads d'analyse en parallèle pour améliorer la rapidité (latence de la première description).
- **Fluidifier l'expérience** globale (transitions, enchaînements, ressenti).
- **Personnalisation du guide audio** (voix, ton, longueur, langue, centres d'intérêt).
- **Tester un modèle local** (vision et/ou TTS) pour réduire latence/coût/dépendance réseau.
- **Ouverture future : portage sur téléphone.**
**Décisions :** le **mouvement (IMU)** devient le déclencheur de référence pour la pause/reprise, à la place de la sortie du champ de vision ; on assume des descriptions plus longues quitte à augmenter le coût TTS, la reprise au point d'arrêt rendant l'écoute fractionnable.

---
## 2026-06-25 — Daniel — Arrêt propre Ctrl+C + run.sh autonome + détection IP WiFi

### 1. Arrêt propre Ctrl+C dans `main.py`

Gros nettoyage de `main.py` : suppression d'un long bloc de code commenté (ancienne tentative d'init du streaming directement depuis notre code — le workaround `streaming_start.py` séparé est désormais assumé).

Sur le fond : ajout d'une fonction `stop_aria_streaming()` qui appelle `aria streaming stop` via subprocess à la sortie. Correction d'un crash C++ (`terminate called without an active exception`) qui apparaissait à chaque Ctrl+C — la cause était que Python tuait le thread Aria brutalement. Résolu en stockant une référence au `streaming_client` dans `_aria_client[]` dès après le `subscribe()`, puis en appelant `unsubscribe()` explicitement dans le `finally` avant de quitter. Les workers vision et TTS sont passés en **daemon threads** pour ne plus bloquer la sortie. Le `finally` garantit dans l'ordre : `quit_audio()` → `unsubscribe()` → `streaming stop`, sur tout chemin de sortie y compris Ctrl+C. Le `__main__` est lui aussi wrappé pour attraper un Ctrl+C pendant l'init.

### 2. `run.sh` — script complet et autonome

Le script existant ne faisait que lancer `streaming_start.py` puis `main.py`. Refonte pour en faire un script de lancement zéro-friction :

- **Détection OS** (`uname -s`) et fonction `ping_once()` pour absorber la différence de flags `ping` entre Linux et macOS.
- **Vérifications de dépendances avec installation automatique** : Python (venv warning si absent, exit si pas de python du tout), `adb` (apt-get sur Linux, brew sur macOS si disponible), `aria` CLI (exit avec message clair si absent), `net-tools` pour `arp` (Linux uniquement, macOS l'a en natif), `streaming_start.py` (vérification que le fichier existe avant de l'appeler).
- **Guards** : ADB ne redémarre que si aucun device n'est connecté ; `aria auth pair` ne tourne que si le SDK n'est pas encore authentifié (ne devrait arriver qu'à la première utilisation sur une nouvelle machine).
- **`STREAM_WAIT=10`** : le sleep post-streaming est maintenant une variable nommée en haut du bloc, facile à ajuster.

### 3. Détection automatique de l'IP Aria en WiFi

Ajout d'une détection d'IP par scan ARP filtré sur le préfixe MAC des lunettes (`2c:26:17`). Logique : on tente d'abord l'IP mise en cache dans `.aria_last_ip` (ping pour vérifier), et on ne scanne l'ARP que si elle est injoignable. L'IP trouvée est sauvegardée pour la session suivante. Fonctionne sur Linux et macOS.

**Fait :** crash C++ à la sortie corrigé ; Ctrl+C propre sans traceback ; run.sh autonome qui s'installe et se configure tout seul ; détection IP WiFi automatique par ARP avec cache.
**Bloqué :** —
**Prochain :**
- Tester le run.sh sur le Mac d'Arthur (différences potentielles à remonter).
- Confirmer que l'authentification SDK persiste bien d'un jour à l'autre.

---
## 2026-06-24 — Daniel — WiFi fonctionnel + arrêt/reprise audio

### 1. Streaming WiFi opérationnel — plus besoin du câble USB

Grande avancée : les lunettes Aria fonctionnent désormais en streaming **WiFi**, sans câble USB. La configuration retenue : connecter les lunettes au **hotspot du téléphone**, avec l'ordinateur et la VM également connectés à ce même hotspot. Le flux arrive correctement sur la VM et le pipeline tourne normalement.

**Point d'attention** : l'IP des lunettes peut changer d'une session à l'autre. Il faut vérifier l'IP assignée aux lunettes avant chaque lancement — à fixer dans la config ou dans `run.sh` pour éviter d'avoir à le retrouver manuellement à chaque fois.

### 2. Arrêt et reprise audio sur la même œuvre

Implémentation de la logique d'arrêt/reprise : quand l'utilisateur détourne le regard d'une œuvre puis revient dessus, l'audio reprend là où il s'était arrêté (plutôt que de recommencer depuis le début ou de relancer une nouvelle génération complète).

**Bug identifié** : dans certains cas, l'explication repart quand même depuis le début au lieu de reprendre à mi-parcours. Cause non encore isolée — il faut instrumenter le debug pour logger quelles phrases sont générées, à quel moment, et à quel index on reprend. Suspicion : le `seen_artworks` / la logique de reprise ne restaure pas correctement l'index de phrase au moment du retour sur l'œuvre.

### 3. Agrandissement des queues

Les queues ont été agrandies. Objectif : tant que l'utilisateur regarde une œuvre, lui fournir des explications en continu sans que le pipeline soit throttlé par une queue trop petite. La contrainte inverse de la session précédente (où on avait *réduit* les queues pour éviter le lag) — ici on parie sur le fait que l'utilisateur reste devant une œuvre suffisamment longtemps pour que des phrases plus nombreuses en attente soient utiles plutôt que périmées.

**Fait :** streaming WiFi via hotspot téléphone opérationnel ; arrêt/reprise audio sur même œuvre implémenté ; queues agrandies.
**Bloqué :** bug de reprise depuis le début au lieu du bon index — à débugger en loggant les phrases générées et les timestamps de reprise.
**Prochain :**
- Isoler le bug de reprise : ajouter des logs sur l'index de phrase sauvegardé au stop et rechargé au retour.
- Fixer l'IP des lunettes dans la config / `run.sh` pour éviter de la chercher manuellement à chaque session WiFi.
- Valider que les queues plus grandes n'introduisent pas de décalage perceptible (re-tester la fluidité perçue).
**Décisions :** on assume le hotspot téléphone comme setup WiFi de référence pour les tests ; on privilégie la continuité des explications (queues larges) sur la fraîcheur des frames pour les sessions longues devant une œuvre.


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
