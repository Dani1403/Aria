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
 L'`END` est aussi exclu de la sauvegarde dans `seen_artworks` (comme le `GENERIC`), pour ne pas polluer la liste des phrases à reprendre plus tard.

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
