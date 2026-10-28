# Stack technique, modèles et outils

## 1. Langage et environnement

- **Python 3.10 – 3.12**
- Environnement virtuel `venv` (ou `conda`)
- VS Code + extensions : Python, Pylance, Jupyter, GitHub Copilot / Copilot Chat

## 2. Choix recommandé : InsightFace (tout-en-un)

| Étape | Modèle (pack `buffalo_l`) | Rôle |
|---|---|---|
| Détection | **SCRFD-10G** (RetinaFace en alternative) | Boîtes + 5 points de repère (yeux, nez, bouche) |
| Alignement | Transformation de similarité sur 5 points → crop 112×112 | Normalise pose, échelle, rotation |
| Embedding | **ArcFace ResNet-100** (entraîné sur Glint360K/WebFace) | Vecteur 512-d normalisé L2 |
| Exécution | `onnxruntime` (CPU suffisant pour 30 images) | Inférence |

Pourquoi ce choix : précision état de l'art, détection + alignement + embedding en un seul appel `FaceAnalysis`, fonctionne sur CPU, poids téléchargés automatiquement au premier lancement.

```python
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
faces = app.get(img_bgr)          # liste de Face : bbox, kps, det_score, normed_embedding
```

## 3. Alternatives (pour comparaison dans le rapport)

| Option | Détection | Embedding | Avantages | Limites |
|---|---|---|---|---|
| **A. InsightFace** (défaut) | SCRFD | ArcFace 512-d | Meilleure précision, simple | Licence des poids : usage non commercial |
| **B. facenet-pytorch** | MTCNN | InceptionResnetV1 (VGGFace2) 512-d | 100 % PyTorch, pédagogique | Un peu moins précis |
| **C. DeepFace** | RetinaFace / MTCNN / OpenCV | ArcFace, Facenet512, VGG-Face | API unifiée pour comparer plusieurs modèles | Plus lent, wrapper |
| **D. dlib / face_recognition** | HOG / CNN | ResNet 128-d | Très simple | Compilation dlib pénible, précision moindre |
| **Baseline** | Haar Cascade OpenCV | LBPH | Référence « classique » à battre | Faible robustesse |

> Recommandation : implémenter **A** comme pipeline principal et **B** en second backend (même interface `FaceEmbedder`) pour une comparaison chiffrée.

## 4. Comparaison et décision

- **Métrique** : similarité cosinus. Les embeddings étant normalisés L2, `cos = a · b` (produit scalaire).
- **Seuil initial** (ArcFace) : `0.40` (à calibrer, plage typique 0.30 – 0.50). Distance cosinus = `1 - cos`.
- **Calibration** : balayer le seuil sur un jeu étiqueté et choisir celui qui maximise le F1 (ou l'EER).
- **Gestion des doublons d'identité** (clé du sujet) :
  1. Une image de la galerie est retenue si **au moins un** de ses visages dépasse le seuil avec la requête.
  2. Optionnel – *query expansion* : moyenner l'embedding requête avec les meilleurs résultats puis relancer la recherche (augmente le rappel).
  3. Optionnel – *clustering de la galerie* (Agglomerative, `metric="cosine"`, `distance_threshold≈0.6`) : toute image appartenant au cluster de la requête est retournée.

## 5. Bibliothèques

| Usage | Bibliothèque |
|---|---|
| Vision | `opencv-python`, `Pillow` |
| Modèles | `insightface`, `onnxruntime` |
| Calcul | `numpy`, `scikit-learn` (clustering, métriques) |
| Alternative B | `torch`, `torchvision`, `facenet-pytorch` |
| Visualisation | `matplotlib`, `seaborn` |
| CLI / config | `argparse` (ou `typer`), `pyyaml` |
| Interface (bonus) | `streamlit` (ou `gradio`) |
| Qualité | `pytest`, `ruff`, `black`, `tqdm`, `loguru` |

## 6. `requirements.txt`

```text
numpy>=1.26
opencv-python>=4.9
Pillow>=10.0
insightface>=0.7.3
onnxruntime>=1.17
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
pyyaml>=6.0
tqdm>=4.66
loguru>=0.7
streamlit>=1.32
pytest>=8.0
ruff>=0.4
# Backend alternatif (optionnel)
# torch>=2.2
# torchvision>=0.17
# facenet-pytorch>=2.5
```

> Sous Windows, `insightface` nécessite parfois les *Microsoft C++ Build Tools*. Si l'installation échoue, utiliser le backend B (`facenet-pytorch`) ou `pip install insightface` depuis un environnement conda.

## 7. Métriques d'évaluation

- **Précision**, **Rappel**, **F1** par requête, puis moyennes sur toutes les requêtes.
- **Courbe ROC** et **EER** sur paires positives/négatives.
- **Matrice de similarité** galerie × galerie (heatmap) pour visualiser les groupes d'identité.
- Temps d'inférence par image.

## 8. Jeux de données de test

- Les ~30 images fournies par le projet + fichier `labels.csv` annoté à la main (`filename,person_id`).
- Bonus de validation : sous-ensemble de **LFW** (Labeled Faces in the Wild) pour tester sur plus de personnes.
