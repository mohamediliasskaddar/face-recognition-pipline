# Refactor – Sujet G2 : tri automatique des visages par personne

> À utiliser avec Copilot : *« Lis docs/REFACTOR_G2.md et suis les étapes R0 à R8 dans l'ordre. Nous en sommes à l'étape Rn. »*
> Ce document **complète** `docs/ROADMAP.md` : le pipeline existant (détection → alignement → embeddings) est conservé, seule la logique de recherche change.

## 1. Ce qui change

| | Ancien sujet | Sujet G2 |
|---|---|---|
| Entrée | 1 image requête + galerie | **Un seul dossier** d'images, pas de requête |
| Question | « Qui ressemble à cette personne ? » | « Combien de personnes y a-t-il et qui est qui ? » |
| Méthode | Comparaison requête ↔ galerie + seuil | **Clustering non supervisé** de tous les visages |
| Unité de travail | L'image | **Le visage** (une image avec 3 visages = 3 éléments) |
| Sortie | Liste d'images correspondantes | **Sous-dossiers `person_01/`, `person_02/`…** contenant les visages recadrés |

Idée centrale : on garde exactement les mêmes embeddings, mais au lieu de comparer à une requête, on **regroupe** les embeddings entre eux.

## 2. Ce qu'on réutilise tel quel

| Module existant | Statut |
|---|---|
| `detection.py` (détection + alignement) | Réutilisé tel quel |
| `embedding.py` | Réutilisé tel quel |
| `gallery.py` (indexation + cache) | Réutilisé (renommer en `indexing.py` si souhaité) ; doit conserver **tous** les visages avec leur `bbox` |
| `io_utils.py`, `config.py`, `visualization.py` | Réutilisés (ajouter des fonctions) |
| `matching.py` (`cosine_similarity_matrix`) | Réutilisé pour la matrice de distances |
| `identity.py` (`cluster_gallery`) | **Devient le cœur** du nouveau projet → déplacé dans `clustering.py` |
| `pipeline.py` / `main.py` | À réécrire (nouveau flux) |
| `evaluation.py` | À adapter (métriques de clustering) |

Ancien mode « recherche par requête » : le conserver en option (`--mode search`), il ne coûte rien et peut servir de bonus en démo.

## 3. Nouvelle arborescence (ajouts en gras)

```text
src/
├── main.py                 # modifié : --input / --output / --mode
├── pipeline.py             # modifié : run_cluster() + run_search() (ancien)
├── detection.py            # inchangé
├── embedding.py            # inchangé (+ option flip, voir R3)
├── gallery.py              # inchangé (indexation + cache)
├── matching.py             # inchangé
├── clustering.py           # NOUVEAU (ex identity.py) : regroupement des visages
├── quality.py              # NOUVEAU : score de qualité (flou, taille, occlusion)
├── cropping.py             # NOUVEAU : recadrage des visages avec marge
├── organizer.py            # NOUVEAU : création des sous-dossiers + export
├── visualization.py        # + planche de contact par personne
└── evaluation.py           # modifié : ARI, NMI, B-cubed
```

## 4. Dossier de sortie

```text
output/
├── person_01/
│   ├── img_03_face0.jpg        # visage recadré
│   ├── img_07_face1.jpg
│   └── ...
├── person_02/
├── person_03/
├── unknown/                    # visages de mauvaise qualité ou non rattachés
├── contact_sheets/             # planche par personne (vérification visuelle)
└── report.json                 # nb de personnes, visages par personne, source de chaque crop
```

Règle de nommage : `<nom_image_source>_face<index>.jpg`. Une image contenant 2 personnes produit **deux crops dans deux dossiers différents**. Option `--copy-originals` pour copier aussi l'image complète dans le dossier de chaque personne.

## 5. Étapes de refactor

### R0 – Préparer
- Créer une branche : `git checkout -b feat/clustering`. (i already created it and m on it right now)
- Garder l'ancien code fonctionnel (ne rien supprimer pour l'instant).
- Mettre à jour `labels.csv` : une ligne **par visage** (`filename,face_index,person_id`), ou par image si chaque image ne contient qu'un visage. Sert uniquement à l'évaluation.

**Done quand** : la branche existe et l'ancien mode tourne encore.

### R1 – Recadrage des visages (`cropping.py`)
- `crop_face(img, bbox, margin=0.25) -> np.ndarray` : recadre avec marge (limites de l'image respectées), carré si possible.
- Option : sauvegarder aussi le visage **aligné** 112×112, mais le crop avec marge est plus lisible pour l'utilisateur.

**Prompt Copilot**
> Crée src/cropping.py avec crop_face(img, bbox, margin) qui ajoute une marge relative, clippe aux bords de l'image et retourne le crop BGR. Ajoute des tests pytest avec bbox en bord d'image.

**Done quand** : les crops de 5 images s'affichent correctement.

### R2 – Qualité des visages (`quality.py`)
Pour gérer occlusions, flou et petits visages, calculer pour chaque visage :
- `det_score` (confiance du détecteur)
- taille de la bbox
- netteté : variance du Laplacien sur le crop (`cv2.Laplacian(gray, cv2.CV_64F).var()`)

`quality_score(face, cfg) -> float` et `is_reliable(face, cfg) -> bool`.
Le score est le minimum des rapports entre chaque mesure et son seuil configuré
(confiance, plus petite dimension de la bbox, netteté) ; un score `>= 1.0`
signifie que les trois critères sont atteints.

Pourquoi : les visages de mauvaise qualité produisent des embeddings instables qui **polluent** les clusters. On les met de côté pendant le clustering, puis on tente de les rattacher (R4).

**Done quand** : on peut lister les visages « fiables » et « douteux » d'un dossier.

### R3 – Robustesse pose / éclairage / expression (`embedding.py`)
Améliorations simples et efficaces :
1. **Flip** : embedding = moyenne normalisée de `embed(face)` et `embed(flip_horizontal(face))`. Aide pour les poses légèrement de profil.
2. **Éclairage** : CLAHE sur le canal L (espace LAB) avant l'embedding, activable via config (à comparer avec/sans).
3. Garder le modèle ArcFace (déjà robuste à l'expression).

**Prompt Copilot**
> Ajoute dans embedding.py une option `use_flip` qui moyenne l'embedding du visage et de son miroir puis renormalise L2, et une option `use_clahe` qui applique un CLAHE sur le canal L en LAB avant l'inférence. Les deux options sont lues depuis config.yaml.

**Done quand** : comparaison chiffrée avec/sans (étape R7).

### R4 – Clustering (`clustering.py`) – cœur du refactor
Le nombre de personnes est **inconnu** : on ne peut pas utiliser K-Means. Utiliser un clustering par seuil de distance.

**Algorithme conseillé**
1. Prendre les embeddings des visages **fiables**.
2. `AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.6)` (seuil à calibrer, plage 0.45 – 0.70).
3. Calculer le **centroïde** de chaque cluster (moyenne normalisée L2).
4. **Rattachement** des visages douteux : similarité cosinus au centroïde le plus proche ; rattaché si `cos >= attach_threshold` (≈ 0.35), sinon → `unknown`.
5. **Fusion** : si deux centroïdes ont `cos >= merge_threshold` (≈ 0.55), fusionner (corrige une personne coupée en 2 par la pose).
6. Option : clusters de taille 1 → les garder comme personne à part entière (une personne apparaît parfois une seule fois) ; ne pas les mettre dans `unknown`.

**Alternatives à tester** : `DBSCAN(metric="cosine", eps=0.5, min_samples=1)` ou `HDBSCAN`. Comparer dans le rapport.

**Prompt Copilot**
> Crée src/clustering.py avec : cluster_faces(embeddings, distance_threshold) via AgglomerativeClustering cosinus average linkage ; compute_centroids ; attach_low_quality(embeddings_low, centroids, threshold) ; merge_close_clusters(centroids, labels, merge_threshold). Retourne un tableau de labels (-1 pour unknown). Ajoute des tests avec des embeddings synthétiques (3 groupes gaussiens).

`compute_centroids` retourne un dictionnaire `label -> centroïde L2 normalisé`.
`attach_low_quality` retourne un label par embedding, avec `-1` si le meilleur
cosinus est sous le seuil. `merge_close_clusters` retourne les centroïdes
fusionnés et les labels remappés ; les fusions transitives sont permises.

**Done quand** : sur des embeddings synthétiques de 3 groupes, on retrouve 3 clusters.

### R5 – Organisation des dossiers (`organizer.py`)
- `organize(records, labels, output_dir, copy_originals=False)` :
  - crée `person_XX/` (numérotation par taille décroissante : `person_01` = la personne la plus fréquente) et `unknown/`
  - sauvegarde chaque crop (`crop_face`) avec le nom `<image>_face<i>.jpg`
  - écrit `report.json`
- Accepte `margin=0.25` et `overwrite=False`. Un dossier de sortie non vide est
  refusé par défaut ; `overwrite=True` le remplace après génération réussie pour
  éviter les résidus d'une exécution précédente.

**Prompt Copilot**
> Crée src/organizer.py avec organize(records, labels, output_dir, copy_originals). Les clusters sont numérotés par taille décroissante (person_01 = le plus grand), label -1 va dans unknown/. Écris report.json avec le nombre de personnes et la liste des crops par dossier.

**Done quand** : l'arborescence de la section 4 est générée.

### R6 – Nouveau pipeline et CLI
`pipeline.run_cluster(input_dir, output_dir, cfg)` :

```text
list_images → detect (tous les visages) → quality filter → align → embed (+cache)
→ cluster (fiables) → attach / merge → crop → organize → contact sheets → report
```

CLI :

```bash
python -m src.main --mode cluster --input data/gallery --output output/
python -m src.main --mode search --query data/query/p.jpg --gallery data/gallery   # ancien mode conservé
```

Options : `--threshold`, `--min-face-size`, `--margin`, `--copy-originals`, `--overwrite`, `--backend`.

Implémentation : `run_cluster` réutilise l'index/cache et exporte aussi les
planches de contact à partir du chemin de rapport retourné par l'organiseur.
Le mode cluster parcourt aussi les sous-dossiers pour traiter les images importées
avec leur arborescence relative; le mode historique search reste non récursif.
Le mode CLI `search` reste le défaut pour préserver les appels historiques;
utiliser explicitement `--mode cluster` pour le nouveau flux.

**Done quand** : une commande produit les dossiers `person_XX/` à partir d'un dossier d'images.

### R7 – Évaluation (`evaluation.py`)
Comme il n'y a plus de requête, on évalue la **qualité du regroupement** avec la vérité terrain `labels.csv` :

| Métrique | Interprétation |
|---|---|
| **Pureté** | Chaque dossier contient-il une seule personne ? |
| **ARI** (Adjusted Rand Index) | Accord global entre clusters et vérité (1 = parfait) |
| **NMI** | Information partagée clusters/vérité |
| **B-cubed précision / rappel / F1** | Précision : pas de mélange de personnes ; rappel : pas de personne éclatée |
| **Nb de personnes détectées vs réel** | Sur- ou sous-segmentation |

À faire : balayer `distance_threshold` de 0.40 à 0.80 et tracer ARI / B-cubed F1 ; choisir le meilleur seuil ; comparer les variantes (sans/avec flip, sans/avec CLAHE, Agglomerative vs DBSCAN, InsightFace vs FaceNet). Analyser visuellement les erreurs (visages mal placés).

**Prompt Copilot**
> Adapte evaluation.py : charge labels.csv (par visage), calcule pureté, ARI, NMI (sklearn.metrics) et B-cubed précision/rappel/F1, puis balaie distance_threshold et trace les courbes avec matplotlib.

Implémentation : `load_face_labels`, `evaluate_clustering`,
`sweep_clustering_threshold` et `plot_clustering_threshold_sweep` exposent ces
étapes. `face_index` est zéro-based et suit l'ordre des visages indexés dans
chaque image; le format historique `filename,person_id` correspond à l'index 0.
Les seuils par défaut balayent `[0.40, 0.80]`.
Sur le jeu courant (29 visages annotés), le balayage donne ARI, NMI et B-cubed
F1 à `1.0` sur toute cette plage, avec 17 personnes détectées pour 18 annotées.
Il n'y a donc pas de seuil optimal unique sur ce jeu; la configuration conserve
`0.60` en attendant une galerie plus représentative, notamment multi-visages.

### R8 – Tests, démo, rapport
- Tests : dossier vide, image sans visage, image avec plusieurs visages, un seul visage au total, deux personnes très proches.
- Planche de contact par personne (`visualization.make_contact_sheet`) pour la démo.
- Mettre à jour `README.md` et `reports/rapport.md` (insister sur la gestion de la pose, l'éclairage, l'expression et l'occlusion : flip, CLAHE, filtrage qualité, rattachement des visages douteux, fusion de clusters).

## 7.Pièges courants

- **Un seul seuil pour tout** : le seuil du mode recherche (0.40 cosinus similarité) n'est pas le même que `distance_threshold` du clustering (distance = 1 − similarité). Ne pas les confondre.
- **Chaînage** : `linkage="single"` fusionne des personnes différentes en chaîne ; préférer `average`.
- **Visages de petite taille ou masqués** : sans filtre qualité, ils créent de faux clusters « petits visages ».
- **Cache** : si l'on change `min_face_size` ou `use_flip`, invalider le cache des embeddings.
- **Nommage** : toujours garder la trace image source → visage → dossier dans `report.json` (utile pour déboguer et pour la démo).