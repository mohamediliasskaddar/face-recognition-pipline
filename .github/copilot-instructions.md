# Instructions pour GitHub Copilot

## Contexte du projet
Pipeline Python de **détection et reconnaissance faciale multi-instances** (module Computer Vision) : à partir d'une image requête, retrouver toutes les images d'une galerie locale (~30 images) contenant la même personne.

Documents de référence (à lire avant toute modification) :
- `README.md` – vue d'ensemble
- `docs/ARCHITECTURE.md` – modules, interfaces, arborescence
- `docs/TECH_STACK.md` – modèles, seuils, dépendances
- `docs/ROADMAP.md` – étapes à suivre dans l'ordre

## Façon de travailler
- Suivre `docs/ROADMAP.md` **étape par étape** ; ne pas implémenter les étapes suivantes sans demande.
- Au début d'une tâche, annoncer l'étape concernée et les fichiers à modifier.
- À la fin, rappeler le critère « Done quand » de l'étape et indiquer comment le vérifier.
- Si une décision d'architecture manque, proposer 2 options brèves plutôt que de choisir silencieusement.

## Stack imposée
- Python 3.10+, `numpy`, `opencv-python`, `insightface` (pack `buffalo_l`: SCRFD + ArcFace), `onnxruntime` (CPU), `scikit-learn`, `matplotlib`, `pytest`.
- Backend alternatif optionnel : `facenet-pytorch` (MTCNN + InceptionResnetV1) derrière la même interface `FaceEmbedder`.

## Conventions de code
- Type hints partout, docstrings en français (style Google), noms de variables/fonctions en anglais.
- Dataclasses pour les structures (`FaceRecord`, `MatchResult`) dans `src/types.py`.
- Pas de chemins en dur : tout passe par `configs/config.yaml` et `src/config.py`.
- Les images sont lues en **BGR** (OpenCV) ; convertir en RGB uniquement pour l'affichage matplotlib.
- Les embeddings sont **float32, normalisés L2** ; la similarité cosinus = produit scalaire.
- Fonctions pures et testables ; pas de logique métier dans `main.py`.
- Logging avec `loguru`, jamais de `print` dans `src/`.
- Style : `ruff` + `black` (ligne max 100).

## Règles métier importantes
- Une image de galerie peut contenir **plusieurs visages** : son score = max des similarités de ses visages.
- Une même personne peut apparaître dans plusieurs images : gérer via seuil, query expansion et clustering optionnel (`src/identity.py`).
- Le seuil initial est `0.40` (cosinus) et doit être calibré avec `src/evaluation.py`.
- Cas limites obligatoires : requête sans visage, requête multi-visages, image de galerie sans visage, fichier corrompu, aucun match.

## À éviter
- Réentraîner des modèles (les modèles pré-entraînés suffisent).
- Charger le modèle plusieurs fois (le créer une fois et le réutiliser).
- Recalculer les embeddings de la galerie à chaque exécution (utiliser le cache `data/cache/`).
- Committer `data/cache/`, `outputs/`, `.venv/` ou des poids de modèles.
- Ajouter des dépendances non listées dans `docs/TECH_STACK.md` sans le signaler.

## Tests
- `pytest -q` doit passer avant de clore une étape.
- Utiliser de petites images dans `tests/fixtures/` ; ne pas dépendre de la galerie complète pour les tests unitaires.
