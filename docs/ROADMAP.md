# Roadmap d'implémentation

> **Mode d'emploi (humain + Copilot)** : avancer **étape par étape**. Chaque étape contient : objectif, tâches, fichiers, prompt Copilot prêt à l'emploi et critère de validation (« Definition of Done »). Ne passer à l'étape suivante que lorsque la case est cochée.
>
> Prompt de départ à coller dans Copilot Chat : *« Lis README.md, docs/ARCHITECTURE.md et docs/TECH_STACK.md. Nous suivons docs/ROADMAP.md. Nous sommes à l'étape N. Implémente uniquement cette étape. »*

## Suivi global

- [x] Étape 0 – Initialisation du projet
- [ ] Étape 1 – Préparation et exploration des données
- [ ] Étape 2 – Détection des visages
- [ ] Étape 3 – Alignement des visages
- [ ] Étape 4 – Extraction des embeddings
- [ ] Étape 5 – Indexation de la galerie (+ cache)
- [ ] Étape 6 – Comparaison et recherche
- [ ] Étape 7 – Identification exhaustive (doublons d'identité)
- [ ] Étape 8 – Pipeline bout en bout et CLI
- [ ] Étape 9 – Visualisation et export des résultats
- [ ] Étape 10 – Évaluation et calibration du seuil
- [ ] Étape 11 – Tests et robustesse
- [ ] Étape 12 – Interface Streamlit (bonus)
- [ ] Étape 13 – Rapport final et livrables

---

## Étape 0 – Initialisation du projet

**Objectif** : squelette propre et environnement fonctionnel.

**Tâches**
1. Créer l'arborescence décrite dans `docs/ARCHITECTURE.md`.
2. Créer le venv et installer `requirements.txt`.
3. Ajouter `.gitignore` (`.venv/`, `data/cache/`, `outputs/`, `__pycache__/`, `*.onnx`).
4. Initialiser git, premier commit.
5. Créer `configs/config.yaml` et `src/config.py` (chargement YAML → dataclass).

**Fichiers** : `requirements.txt`, `.gitignore`, `pyproject.toml`, `configs/config.yaml`, `src/config.py`, tous les `__init__.py`.

**Prompt Copilot**
> Crée l'arborescence de docs/ARCHITECTURE.md avec des fichiers vides et docstrings. Implémente `src/config.py` qui charge `configs/config.yaml` dans des dataclasses typées.

**Done quand** : `python -c "from src.config import load_config; print(load_config())"` affiche la config.

---

## Étape 1 – Préparation et exploration des données

**Objectif** : connaître la galerie et créer la vérité terrain.

**Tâches**
1. Placer ~30 images dans `data/gallery/`, la requête dans `data/query/`.
2. Implémenter `src/io_utils.py` : `list_images(folder)`, `read_image(path)` (BGR, gestion EXIF orientation), `save_image`.
3. Notebook `01_exploration_data.ipynb` : afficher la galerie en mosaïque, résolutions, nombre approximatif de visages.
4. Créer `data/labels.csv` à la main : `filename,person_id` (une ligne par image, plusieurs `person_id` si plusieurs personnes → utiliser `;` ou plusieurs lignes).

**Prompt Copilot**
> Implémente `src/io_utils.py` avec `list_images` (extensions jpg/jpeg/png/bmp/webp, tri alphabétique), `read_image` qui corrige l'orientation EXIF et lève une erreur claire si le fichier est illisible.

**Done quand** : la mosaïque de la galerie s'affiche et `labels.csv` couvre toutes les images.

---

## Étape 2 – Détection des visages

**Objectif** : localiser tous les visages d'une image.

**Tâches**
1. Dans `src/types.py`, définir `FaceRecord` et `MatchResult`.
2. Dans `src/detection.py`, implémenter `FaceDetector` (InsightFace `FaceAnalysis`, pack `buffalo_l`, `det_size=(640,640)`).
3. Filtrer par `min_det_score` et `min_face_size` (config).
4. Implémenter `select_main_face(faces)` (aire × score).
5. Notebook `02_detection_alignement.ipynb` : dessiner boîtes + 5 landmarks sur quelques images.

**Prompt Copilot**
> Implémente `FaceDetector` dans src/detection.py avec insightface FaceAnalysis. La méthode `detect(img_bgr)` retourne une liste de faces filtrées par score et taille minimale, triées par aire décroissante. Ajoute `select_main_face`.

**Done quand** : détection correcte sur ≥ 90 % des images de la galerie (vérifiée visuellement).

---

## Étape 3 – Alignement des visages

**Objectif** : normaliser pose, échelle et rotation (crop 112×112).

**Tâches**
1. Implémenter `FaceDetector.align(img, landmarks)` avec `insightface.utils.face_align.norm_crop(img, landmark=kps, image_size=112)`.
2. Comparer visuellement visages avant/après alignement dans le notebook.
3. (Option pédagogique) Réimplémenter l'alignement avec `cv2.estimateAffinePartial2D` pour comprendre la transformation de similarité.

**Prompt Copilot**
> Ajoute la méthode `align` qui retourne un crop 112x112 via norm_crop. Écris un test qui vérifie la forme (112,112,3).

**Done quand** : grille de visages alignés (yeux horizontaux, même échelle) générée dans le notebook.

---

## Étape 4 – Extraction des embeddings

**Objectif** : projeter chaque visage aligné dans un vecteur 512-d normalisé.

**Tâches**
1. Dans `src/embedding.py`, définir `FaceEmbedder(ABC)` avec `embed(aligned) -> np.ndarray`.
2. Implémenter `InsightFaceEmbedder` (ArcFace via `face.normed_embedding`, ou modèle `recognition` direct sur le crop aligné).
3. Vérifier la norme L2 = 1.
4. (Bonus) `FacenetEmbedder` avec `facenet-pytorch` pour comparaison.

**Prompt Copilot**
> Implémente l'interface FaceEmbedder et InsightFaceEmbedder. Les embeddings doivent être normalisés L2 en float32. Ajoute un test sur la dimension (512,) et la norme ≈ 1.

**Done quand** : deux photos de la même personne ont un cosinus nettement supérieur à deux personnes différentes (test sur 3–4 paires).

---

## Étape 5 – Indexation de la galerie (+ cache)

**Objectif** : calculer une seule fois les embeddings de la galerie.

**Tâches**
1. `src/gallery.py` : `build_gallery_index(folder, detector, embedder) -> list[FaceRecord]` (tous les visages de chaque image).
2. `save_cache / load_cache` dans `data/cache/embeddings.npz` + métadonnées (chemins, bbox) ; invalidation si la liste ou la date des fichiers change.
3. Logger les images sans visage.
4. Barre de progression `tqdm`.

**Prompt Copilot**
> Implémente gallery.py : indexation de toutes les images avec plusieurs visages par image, cache npz avec hash du contenu du dossier, rechargement automatique si le cache est valide.

**Done quand** : 2ᵉ exécution quasi instantanée grâce au cache.

---

## Étape 6 – Comparaison et recherche

**Objectif** : scorer la requête contre toute la galerie.

**Tâches**
1. `src/matching.py` : `cosine_similarity_matrix` (produit matriciel), `match_images(query_emb, records, threshold)`.
2. Agréger **par image** : score = max des similarités de ses visages.
3. Trier par score décroissant ; retourner `list[MatchResult]`.
4. Afficher le top-10 avec scores dans le notebook.

**Prompt Copilot**
> Implémente matching.py avec une similarité cosinus vectorisée numpy. `match_images` agrège le score maximal par image, filtre par seuil et trie. Ajoute des tests avec des embeddings synthétiques.

**Done quand** : test unitaire vert + top-10 plausible visuellement.

---

## Étape 7 – Identification exhaustive (doublons d'identité)

**Objectif** : retrouver **toutes** les images de la personne, même celles peu similaires à la requête (profil, lunettes, éclairage…).

**Tâches**
1. **Query expansion** : prendre les `k` meilleurs matches très sûrs (score > seuil + marge), moyenner leurs embeddings avec celui de la requête, renormaliser, relancer la recherche.
2. **Clustering de la galerie** (option) : `AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=0.6, n_clusters=None)` ; retourner toutes les images du cluster de la requête.
3. `resolve_identity(...)` combine les stratégies selon `config.yaml`.
4. Comparer les 3 variantes (seuil seul, +expansion, +clustering) sur le jeu étiqueté.

**Prompt Copilot**
> Implémente identity.py : expand_query, cluster_gallery (sklearn AgglomerativeClustering cosinus) et resolve_identity qui combine seuil, expansion et clustering selon la config. Chaque stratégie doit être activable indépendamment.

**Done quand** : le rappel augmente (ou reste égal) avec l'expansion sans dégrader la précision de façon notable.

---

## Étape 8 – Pipeline bout en bout et CLI

**Objectif** : une commande unique.

**Tâches**
1. `src/pipeline.py` : `run(query_path, gallery_dir, output_dir, cfg)`.
2. `src/main.py` avec `argparse` : `--query`, `--gallery`, `--output`, `--threshold`, `--face-index`, `--backend`, `--no-cache`.
3. Gérer les cas limites (voir `docs/ARCHITECTURE.md` §6) avec messages clairs et logs `loguru`.

**Prompt Copilot**
> Implémente pipeline.run et le CLI main.py. Gère : aucun visage dans la requête, plusieurs visages (option --face-index), galerie vide, aucun match. Retourne un code de sortie non nul en cas d'erreur.

**Done quand** :
```bash
python -m src.main --query data/query/person.jpg --gallery data/gallery --output outputs/
```
produit `outputs/matches/` et `report.json`.

---

## Étape 9 – Visualisation et export des résultats

**Tâches**
1. `src/visualization.py` : `draw_faces(img, records)` (boîte + score), `make_result_grid(query, matches)`, `plot_similarity_heatmap(records)`.
2. Export : copie des images dans `outputs/matches/`, annotées dans `outputs/annotated/`, grille `result_grid.png`, `report.json`.

**Prompt Copilot**
> Implémente visualization.py avec matplotlib/OpenCV : annotation des visages détectés, grille requête + résultats avec scores, et heatmap galerie×galerie.

**Done quand** : la grille de résultats est lisible et exportée.

---

## Étape 10 – Évaluation et calibration du seuil

**Tâches**
1. `src/evaluation.py` : précision, rappel, F1 par requête à partir de `labels.csv`.
2. Utiliser **plusieurs requêtes** (une par personne de la galerie) pour moyenner.
3. `sweep_threshold` : courbe P/R/F1 pour seuils de 0.2 à 0.7 ; choisir le meilleur.
4. ROC et EER sur toutes les paires (même personne vs personnes différentes).
5. Notebook `03_analyse_seuils.ipynb` : tracer les courbes et analyser les erreurs (faux positifs / faux négatifs avec images).

**Prompt Copilot**
> Implémente evaluation.py : lecture de labels.csv, calcul P/R/F1 pour une requête et en moyenne, balayage de seuil, courbe ROC avec EER. Génère les figures avec matplotlib.

**Done quand** : seuil optimal choisi, reporté dans `config.yaml`, et tableau comparatif des variantes disponible.

---

## Étape 11 – Tests et robustesse

**Tâches**
1. Tests `pytest` : détection (forme), embedding (norme), matching (tri/seuil), pipeline (test d'intégration sur 3–4 images).
2. Tester les cas limites : image sans visage, fichier corrompu, galerie vide, requête multi-visages.
3. `ruff check` + `black` ; typage des signatures publiques.

**Prompt Copilot**
> Écris les tests pytest listés dans docs/ROADMAP.md étape 11 en utilisant de petites images de fixtures dans tests/fixtures/. Corrige les bugs détectés.

**Done quand** : `pytest -q` entièrement vert.

---

## Étape 12 – Interface Streamlit (bonus)

**Tâches**
1. `app/streamlit_app.py` : upload de la requête, sélection du dossier galerie, curseur de seuil, bouton « Rechercher ».
2. Affichage de la requête, des résultats en grille avec scores, et bouton de téléchargement ZIP.

**Prompt Copilot**
> Crée app/streamlit_app.py qui appelle pipeline.run, avec uploader d'image, slider de seuil, affichage en grille des résultats et téléchargement ZIP.

**Lancer** : `streamlit run app/streamlit_app.py`

---

## Étape 13 – Rapport final et livrables

**Plan du rapport (`reports/rapport.md`)**
1. Contexte et objectifs
2. Architecture du pipeline (schéma)
3. Choix des modèles et justification (détection, alignement, embedding)
4. Mesure de similarité, seuil et gestion des doublons d'identité
5. Résultats : tableaux P/R/F1, ROC, exemples réussis et échecs
6. Comparaison des backends/variantes (InsightFace vs FaceNet, avec/sans expansion)
7. Limites (pose, éclairage, occlusions, biais démographiques) et perspectives
8. Considérations éthiques (vie privée, usage des données biométriques)

**Livrables**
- Code source propre et documenté
- `README.md` à jour
- Notebooks d'analyse
- Rapport + démonstration (CLI ou Streamlit)

---

## Annexe – Planning indicatif

| Phase | Étapes | Durée estimée |
|---|---|---|
| Fondations | 0 – 1 | 0,5 jour |
| Cœur du pipeline | 2 – 6 | 1,5 jour |
| Identification exhaustive | 7 | 0,5 jour |
| Intégration | 8 – 9 | 0,5 jour |
| Évaluation et qualité | 10 – 11 | 1 jour |
| Bonus + rapport | 12 – 13 | 1 jour |
