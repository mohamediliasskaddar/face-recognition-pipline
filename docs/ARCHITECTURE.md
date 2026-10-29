# Architecture du projet

## 1. Vue d'ensemble

```mermaid
flowchart LR
    Q[Image requête] --> D1[Détection]
    G[Galerie ~30 images] --> D2[Détection]
    D1 --> A1[Alignement 112x112]
    D2 --> A2[Alignement 112x112]
    A1 --> E1[Embedding requête]
    A2 --> E2[Embeddings galerie]
    E2 --> C[(Cache .npz)]
    E1 --> S[Similarité cosinus]
    C --> S
    S --> T[Seuil + agrégation par image]
    T --> R[Regroupement d'identité]
    R --> O[Images correspondantes + rapport JSON]
```

## 2. Arborescence du projet

```text
face-recognition-pipeline/
├── README.md
├── requirements.txt
├── pyproject.toml                 # config ruff/black/pytest
├── .gitignore
├── .github/
│   └── copilot-instructions.md
├── configs/
│   └── config.yaml                # modèle, seuils, chemins
├── data/
│   ├── gallery/                   # ~30 images de référence
│   ├── query/                     # image(s) requête
│   ├── labels.csv                 # vérité terrain (filename,person_id)
│   └── cache/                     # embeddings.npz (généré)
├── docs/
│   ├── ROADMAP.md
│   ├── ARCHITECTURE.md
│   └── TECH_STACK.md
├── notebooks/
│   ├── 01_exploration_data.ipynb
│   ├── 02_detection_alignement.ipynb
│   └── 03_analyse_seuils.ipynb
├── src/
│   ├── __init__.py
│   ├── main.py                    # point d'entrée CLI
│   ├── config.py                  # chargement de config.yaml
│   ├── types.py                   # dataclasses : FaceRecord, MatchResult
│   ├── io_utils.py                # lecture images, listing dossier, sauvegarde
│   ├── detection.py               # FaceDetector (détection + landmarks + alignement)
│   ├── embedding.py               # FaceEmbedder (interface + backends)
│   ├── gallery.py                 # indexation de la galerie + cache
│   ├── matching.py                # similarité cosinus, seuil, agrégation par image
│   ├── identity.py                # regroupement d'identité, query expansion
│   ├── pipeline.py                # orchestration bout en bout
│   ├── visualization.py           # boîtes, grilles de résultats, heatmaps
│   └── evaluation.py              # précision / rappel / F1 / ROC
├── app/
│   └── streamlit_app.py           # interface (bonus)
├── tests/
│   ├── test_detection.py
│   ├── test_embedding.py
│   ├── test_matching.py
│   └── test_pipeline.py
├── outputs/                       # résultats générés (ignoré par git)
└── reports/
    └── rapport.md                 # rapport final (méthode, résultats, analyse)
```

## 3. Responsabilités et contrats d'interface

### `types.py`

```python
@dataclass
class FaceRecord:
    image_path: str
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2
    landmarks: np.ndarray               # (5, 2)
    det_score: float
    aligned: np.ndarray                 # (112, 112, 3) BGR
    embedding: np.ndarray               # (512,) normalisé L2

@dataclass
class MatchResult:
    image_path: str
    best_score: float                   # cosinus max parmi les visages de l'image
    matched_face_index: int
```

### `detection.py`
- `FaceDetector.detect(img) -> list[Face]` : boîtes, 5 points, score.
- `FaceDetector.align(img, landmarks) -> np.ndarray` : crop 112×112 normalisé.
- `select_main_face(faces) -> Face` : choisit le visage de la requête (le plus grand × meilleur score) si plusieurs.

### `embedding.py`
- `class FaceEmbedder(ABC)` avec `embed(aligned_face) -> np.ndarray`.
- `InsightFaceEmbedder` extrait l'embedding ArcFace avec le modèle `recognition`
  de `FaceAnalysis`, puis retourne un vecteur `(512,)` normalisé L2 en `float32`.
- Si `FaceAnalysis.get` a déjà produit `face.normed_embedding`, l'indexeur le
  réutilise pour éviter une deuxième inférence ArcFace sur le même visage.
- Le moteur `FaceAnalysis` peut être partagé avec `FaceDetector` via son argument
  `analysis`, afin d'éviter de charger les modèles deux fois.
- `FacenetEmbedder` est une alternative prévue, mais pas encore implémentée.

### `gallery.py`
- `build_gallery_index(folder, detector, embedder) -> list[FaceRecord]` : traite
  chaque image et crée un enregistrement par visage, même si plusieurs visages
  apparaissent sur la même photo.
- Les fichiers illisibles et images sans visage sont ignorés avec un avertissement;
  la progression est affichée avec `tqdm`.
- `save_cache / load_cache` : `data/cache/embeddings.npz`, contenant embeddings,
  crops et métadonnées sans pickle. Le cache est invalidé si un nom/contenu de
  galerie ou un paramètre de détection/embedding change.

### `matching.py`
- `cosine_similarity_matrix(query_emb, gallery_embs) -> np.ndarray`.
- `match_images(query_emb, records, threshold) -> list[MatchResult]` : agrège par image (max sur ses visages), filtre par seuil, trie par score décroissant.
- La matrice accepte une requête `(D,)` ou plusieurs requêtes `(Q, D)` et une galerie `(N, D)`; les vecteurs sont normalisés lors du calcul. Les embeddings vides, non finis, de norme nulle ou de dimensions incompatibles produisent une erreur explicite.
- `match_images` retourne l'index zéro-based du visage gagnant au sein de son image; le seuil est inclusif.

### `identity.py`
- `expand_query(query_emb, top_matches, records, k) -> np.ndarray` : moyenne normalisée de la requête et des embeddings des meilleures correspondances.
- `cluster_gallery(records, distance_threshold) -> labels` : produit une étiquette par visage avec AgglomerativeClustering, distance cosinus et linkage moyen.
- `resolve_identity(query_emb, records, cfg) -> list[MatchResult]` : démarre par le seuil puis ajoute les résultats de l'expansion et/ou du cluster selon les interrupteurs de configuration. L'expansion exige un score strictement supérieur à `threshold + 0.05`; le clustering insère la requête comme premier vecteur pour sélectionner son cluster.
- Chaque stratégie peut être désactivée indépendamment via `matching.use_query_expansion` et `matching.use_gallery_clustering`. Les stratégies ajoutent des candidats sans retirer les résultats initiaux.

### `pipeline.py`
- `run(query_path, gallery_dir, output_dir, cfg, *, face_index=None, use_cache=True) -> list[MatchResult]`.
- Le pipeline partage une seule instance InsightFace entre détection et embedding, sélectionne le visage principal par défaut ou le visage zéro-based demandé, construit/recharge l'index galerie, appelle `resolve_identity`, copie les résultats dans `matches/` et écrit `report.json`.
- Un dossier de galerie sans image ou sans visage indexable et une requête sans visage sont des erreurs explicites. Une recherche sans correspondance est un résultat valide : `matches/` et un rapport avec une liste vide sont produits.

### `evaluation.py`
- `evaluate(results, labels_csv, query_person_id) -> dict` : P, R, F1.
- `sweep_threshold(...)` : courbe P/R/F1 selon le seuil.
- `roc_curve_pairs(...)` : ROC + EER.

## 4. `configs/config.yaml` (exemple)

```yaml
model:
  backend: insightface          # insightface | facenet
  name: buffalo_l
  det_size: [640, 640]
  providers: [CPUExecutionProvider]

detection:
  min_det_score: 0.5
  min_face_size: 40

matching:
  threshold: 0.40               # similarité cosinus
  use_query_expansion: true
  expansion_top_k: 3
  use_gallery_clustering: false
  cluster_distance_threshold: 0.60

paths:
  cache_dir: data/cache
  output_dir: outputs
```

## 5. Format de sortie

`outputs/` contient :

```text
outputs/
├── matches/                       # copies des images correspondantes
└── report.json
```

`report.json` :

```json
{
  "query": "data/input/query.jpg",
  "threshold": 0.40,
  "n_gallery_images": 30,
  "n_matches": 2,
  "matches": [
    {
      "image": "data/gallery/img_03.jpg",
      "score": 0.71,
      "matched_face_index": 0
    }
  ]
}
```

`annotated/` et `result_grid.png` relèvent de l'étape 9 (visualisation), pas de
l'export minimal de l'étape 8.

## 6. Cas limites à gérer

| Cas | Comportement attendu |
|---|---|
| Aucun visage dans la requête | Erreur claire, code retour ≠ 0 |
| Plusieurs visages dans la requête | Visage principal par défaut, option `--face-index` |
| Image de galerie sans visage | Ignorée, avertissement loggé |
| Image de galerie avec plusieurs visages | Une image est retenue si au moins un visage correspond |
| Visage petit, flou ou de profil | Filtrage par `min_det_score` / `min_face_size` |
| Aucun résultat au-dessus du seuil | Retourner liste vide + meilleur score observé |
| Fichier non image / corrompu | Ignoré et loggé |
