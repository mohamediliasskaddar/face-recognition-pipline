# Référence de `src/` – rôle et entrées/sorties de chaque étape

> Document de référence unique. Il alimente : (1) la commande `--mode trace` (identifiants `S01`…`S10`), (2) la page `docs/pipeline/index.html`, (3) l'explication orale du projet.
> Si le code d'une étape change, mettre à jour ce fichier **et** `docs/pipeline/stages.json`.

## 1. Vue globale

```text
MODE TRI (clustering)
Dossier d'images ─► S01 Ingestion ─► S02 Détection ─► S03 Qualité ─► S04 Alignement ─► S05 Embedding
   ─► S06 Similarité ─► S07 Clustering ─► S08 Rattachement/Fusion ─► S09 Recadrage ─► S10 Organisation
   ─► output/person_01/, person_02/, …, unknown/ + report.json

MODE RECHERCHE (requête)
Image requête + galerie ─► S01 ─► S02 ─► S04 ─► S05 ─► M1 Matching ─► M2 Identité (expansion/cluster) ─► images correspondantes
```

## 2. Étapes du mode Tri

| ID | Étape | Module `src/` | Entrée | Sortie | Rôle |
|---|---|---|---|---|---|
| S01 | Ingestion | `io_utils.py` | Dossier d'images (jpg/png/bmp/webp) | Liste d'images BGR valides (EXIF corrigé) + liste des fichiers ignorés | Lire et valider les images ; ignorer les fichiers corrompus |
| S02 | Détection | `detection.py` | Image BGR | Liste de visages : boîte `(x1,y1,x2,y2)`, 5 points de repère, score de confiance | Localiser **tous** les visages d'une image (SCRFD) |
| S03 | Qualité | `quality.py` | Visages détectés + image | Visages « fiables » / « douteux » avec score (confiance, taille, netteté) | Écarter les visages flous, petits ou occultés avant le clustering |
| S04 | Alignement | `detection.py` (`align`) | Image + 5 points de repère | Visage normalisé 112×112 (yeux horizontaux, même échelle) | Neutraliser rotation et échelle pour comparer des visages équivalents |
| S05 | Embedding | `embedding.py` | Visage aligné 112×112 | Vecteur 512-d, float32, norme L2 = 1 (option flip/CLAHE) | Représenter l'identité du visage dans un espace où « même personne = proche » |
| S06 | Similarité | `matching.py` | Matrice N×512 d'embeddings | Matrice N×N de similarités cosinus | Mesurer la ressemblance entre tous les visages |
| S07 | Clustering | `clustering.py` | Embeddings des visages fiables + `distance_threshold` | Label de cluster par visage (nombre de personnes inconnu) | Regrouper les visages d'une même personne (agglomératif, distance cosinus) |
| S08 | Rattachement / Fusion | `clustering.py` | Visages douteux + centroïdes des clusters | Labels finaux (`-1` = unknown) ; clusters trop proches fusionnés | Récupérer les visages difficiles et corriger une personne coupée en deux |
| S09 | Recadrage | `cropping.py` | Image source + boîte du visage | Crop du visage avec marge (par défaut 25 %) | Produire l'image de visage à ranger |
| S10 | Organisation | `organizer.py` | Crops + labels finaux | Dossiers `person_01/`, `person_02/`…, `unknown/`, `report.json` | Ranger les visages par personne (numérotation par taille décroissante) |

## 3. Étapes du mode Recherche (ancien sujet, conservé)

| ID | Étape | Module `src/` | Entrée | Sortie | Rôle |
|---|---|---|---|---|---|
| M0 | Indexation galerie | `gallery.py` | Dossier galerie | Embeddings de tous les visages + cache `data/cache/*.npz` | Calculer une seule fois les embeddings |
| M1 | Matching | `matching.py` | Embedding de la requête + embeddings galerie + seuil | Liste `MatchResult` (image, score max) triée | Comparer la requête à chaque visage ; score d'une image = max de ses visages |
| M2 | Identité | `identity.py` | Résultats M1 + embeddings | Liste élargie d'images de la même personne | Query expansion (moyenne des meilleurs matches) pour retrouver les occurrences difficiles |

## 4. Modules transverses (sans étape dédiée)

| Module | Rôle |
|---|---|
| `config.py` | Charge `configs/config.yaml` en dataclasses typées |
| `types.py` | Structures `FaceRecord`, `MatchResult` |
| `pipeline.py` | Orchestration : `run_cluster()` (tri), `run_search()` (recherche) ; accepte un `Tracer` optionnel |
| `main.py` | CLI : `--mode cluster | search | trace` |
| `visualization.py` | Boîtes sur les images, grilles de résultats, heatmaps, planches de contact |
| `evaluation.py` | Métriques : pureté, ARI, NMI, B-cubed (tri) ; précision/rappel/F1 (recherche) |
| `trace.py` | (V4) Enregistre entrée/sortie/durée de chaque étape et produit `trace.json` + `trace_report.md` |

## 5. Commandes utiles

```bash
# Tri complet d'un dossier (mode principal)
python -m src.main --mode cluster --input data/gallery --output output/ --overwrite

# Recherche des images d'une personne (ancien mode)
python -m src.main --mode search --query data/query/person.jpg --gallery data/gallery --output output/

# Trace pas à pas du pipeline (une image ou un dossier)
python -m src.main --mode trace --input data/gallery/img_03.jpg --output outputs/trace/
python -m src.main --mode trace --input data/gallery --output outputs/trace/ --max-images 10

# Interface web
streamlit run app/streamlit_app.py

# Évaluation du clustering (nécessite data/labels.csv)
python -m src.evaluation --input data/gallery --labels data/labels.csv --sweep

# Tests et qualité
pytest -q
ruff check . && black --check .
```

## 6. Exemple de lecture « de bout en bout » (pour la présentation orale)

> *« Une photo de groupe entre dans le pipeline. **S02** détecte 4 visages. **S03** en retient 3 de bonne qualité et met 1 visage flou de côté. **S04** redresse et normalise les 3 visages en 112×112. **S05** transforme chacun en un vecteur de 512 nombres. **S07** voit que 2 vecteurs sont très proches : c'est la même personne, ils vont dans le même groupe. **S08** rattache le visage flou au groupe dont le centre est le plus proche. **S09** recadre les 4 visages, **S10** les range dans `person_01/`, `person_02/` et `person_03/`. »*

## 7. Paramètres clés par étape (`configs/config.yaml`)

| Étape | Paramètre | Effet |
|---|---|---|
| S02 | `det_size`, `min_det_score` | Sensibilité de la détection |
| S03 | `min_face_size`, `min_sharpness` | Sévérité du filtre qualité |
| S05 | `use_flip`, `use_clahe` | Robustesse pose / éclairage |
| S07 | `distance_threshold` | Plus bas = plus de personnes (sur-segmentation) ; plus haut = personnes fusionnées |
| S08 | `attach_threshold`, `merge_threshold` | Rattachement des visages douteux / fusion des clusters |
| S09 | `crop_margin` | Marge autour du visage |
