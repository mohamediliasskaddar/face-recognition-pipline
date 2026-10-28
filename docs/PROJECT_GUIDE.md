# Guide de suivi du projet

Ce document explique les composants déjà préparés, leur rôle dans le pipeline et
les étapes restantes. Il complète la [roadmap](./ROADMAP.md) sans remplacer ses
critères de validation.

## But du projet

À partir d'une photo requête, retrouver dans une galerie locale toutes les images
qui contiennent la même personne. Une image peut contenir plusieurs personnes,
et une même personne peut apparaître dans plusieurs images.

Le pipeline prévu suit ces étapes :

```text
configuration + images
        ↓
lecture des images (BGR)
        ↓
détection des visages et repères
        ↓
alignement des visages
        ↓
embeddings numériques
        ↓
comparaison avec la galerie
        ↓
résultats, visualisations et évaluation
```

Un embedding est une représentation numérique d'un visage. Les embeddings
permettront de comparer les visages sans comparer directement les pixels des
photos, qui peuvent varier en éclairage, pose ou arrière-plan.

## État actuel

| Composant | État | Rôle |
|---|---|---|
| `configs/config.yaml` et `src/config.py` | Préparés | Paramètres du modèle, seuils et chemins chargés dans des dataclasses typées. |
| `src/io_utils.py` | Implémenté | Liste les images, corrige leur orientation EXIF, les lit en BGR et permet de les enregistrer. |
| `src/detection.py` | Étapes 2–3 implémentées | Initialise InsightFace, détecte et filtre les visages, aligne un visage en crop `112 × 112` et sélectionne un visage principal. |
| `src/embedding.py` | Code de l'étape 4 implémenté | Extrait un vecteur ArcFace `(512,)`, normalisé L2 en `float32`, à partir d'un crop aligné. La séparation des identités sur le jeu réel reste à confirmer. |
| `src/types.py` | Étape 5 implémentée | Définit `FaceRecord` pour stocker le chemin, la boîte, les landmarks, le score, le crop aligné et l'embedding, ainsi que `MatchResult`. |
| `src/gallery.py` | Étape 5 implémentée | Construit un enregistrement par visage, ignore les images illisibles/sans visage avec avertissement, et recharge le cache NPZ s'il correspond à la galerie et à la configuration. |
| `src/matching.py` et `src/identity.py` | À compléter | Compareront les vecteurs, agrégeront les scores par image et géreront les identités récurrentes. |
| `src/pipeline.py` et `src/main.py` | À compléter | Relieront les composants pour exécuter une recherche de bout en bout. |
| `src/visualization.py` et `src/evaluation.py` | À compléter | Aideront à examiner les résultats et mesurer précision/rappel. |

Les images de galerie et la photo requête ont été ajoutées aux données du projet.
La création/validation de `labels.csv` et l'exploration visuelle dans le notebook
restent à confirmer avant de considérer toute l'étape de préparation des données
comme terminée.

## Fonctionnement du détecteur

### Création et réutilisation

`FaceDetector` lit la configuration du projet, sauf si une configuration lui est
fournie, puis crée une instance `insightface.app.FaceAnalysis`. Le modèle indiqué
dans la configuration (`buffalo_l` par défaut) est préparé une seule fois à la
création du détecteur. Le même objet détecteur doit ensuite être réutilisé pour
la requête et les images de la galerie, plutôt que de recharger les poids à
chaque image.

La configuration actuelle utilise `CPUExecutionProvider` et une taille de
détection de `640 × 640`. L'initialisation réelle nécessite InsightFace et ses
modèles pré-entraînés installés sur la machine.

### Détection et filtres

`detect(img_bgr)` prend un tableau image BGR `uint8`, comme celui retourné par
`read_image`. Il transmet cette image à InsightFace, puis :

1. écarte les visages dont le score est inférieur à `detection.min_det_score`;
2. écarte les boîtes dont la plus petite dimension est inférieure à
   `detection.min_face_size`;
3. trie les visages restants par aire décroissante.

La méthode retourne les objets visage InsightFace, qui conservent notamment les
boîtes (`bbox`), les points de repère (`kps`) et le score (`det_score`). Les
étapes d'alignement et d'embedding pourront réutiliser ces informations.
Une image invalide ou une boîte incohérente du moteur produit une erreur claire,
pas un résultat de détection présenté comme valide.

### Visage principal

`select_main_face(faces)` sert lorsqu'une seule personne doit être choisie dans
une image requête qui contient plusieurs visages. Le critère de l'architecture
est `aire de la boîte × score de détection`. Une liste vide provoque une erreur
explicite. Ce choix par défaut ne supprime pas les autres visages de la galerie :
ils doivent rester disponibles pour la recherche multi-visages.

### Alignement

`align(img_bgr, landmarks)` utilise les cinq repères fournis par InsightFace
(yeux, nez et coins de la bouche) et `insightface.utils.face_align.norm_crop`
avec `image_size=112`. Le résultat est un tableau BGR `uint8` de forme
`(112, 112, 3)`. Cette transformation remet les visages à une taille et une pose
comparables avant de calculer leurs embeddings.

La méthode vérifie la forme et les valeurs des landmarks ainsi que la forme du
crop retourné. Cela évite de transmettre silencieusement une entrée ou sortie
invalide aux prochaines étapes.

## Extraction des embeddings

`FaceEmbedder` définit l'interface commune; `InsightFaceEmbedder` est son backend
ArcFace. La méthode `embed` prend un crop BGR `uint8` de forme `(112, 112, 3)`,
appelle directement `get_feat` du modèle de reconnaissance et retourne un
vecteur `float32` de 512 valeurs, normalisé L2.

L'extracteur rejette les crops d'une autre forme, les sorties ArcFace mal
dimensionnées ou non finies, et les vecteurs nuls qui ne peuvent pas être
normalisés.

Pour ne charger `FaceAnalysis` qu'une seule fois, partager son instance entre les
deux composants :

```python
from src.detection import FaceDetector
from src.embedding import InsightFaceEmbedder

embedder = InsightFaceEmbedder()
detector = FaceDetector(analysis=embedder.analysis)
```

Le moteur contient alors le détecteur et le modèle ArcFace de `buffalo_l`. Pour
chaque visage, `detector.detect(image)` donne ses repères; `detector.align(image,
face.kps)` produit le crop qui est passé à `embedder.embed(crop)`.

### Tests sans téléchargement des modèles

Les tests unitaires remplacent le moteur InsightFace par un faux détecteur. Ils
vérifient la configuration transmise, les filtres, l'ordre par aire, la forme
du crop aligné, la dimension et la normalisation L2 des embeddings, les entrées
invalides et la sélection du visage principal sans télécharger les poids.
Cela valide la logique Python, mais ne prouve pas encore que les modèles
détectent correctement les visages de la galerie. La validation visuelle avec le
vrai modèle reste nécessaire pour le critère « Done quand » de l'étape 2.

### Vérification réelle des embeddings

Une image de galerie a été passée dans la chaîne réelle détection → alignement →
ArcFace. Le résultat était un embedding `(512,)` en `float32` de norme L2 `1.0`.
Sur cette image, son cosinus avec l'embedding natif normalisé de FaceAnalysis
était aussi `1.0`, ce qui confirme que l'extraction utilise la même représentation
que le modèle InsightFace.

Une vérification exploratoire de 29 images regroupées selon l'identifiant écrit
dans leur nom de fichier n'a toutefois pas montré une séparation nette entre
paires du même groupe et paires de groupes différents. Les noms de fichiers ne
remplacent pas une vérité terrain vérifiée; `data/labels.csv` ne contient encore
que son en-tête. Le code d'extraction est donc fonctionnel, mais le critère
qualitatif de l'étape 4 n'est pas encore confirmé. Il faut vérifier visuellement
les identités et compléter les labels avant de conclure que les embeddings
distinguent correctement ces personnes.

## Indexation de la galerie et cache

`build_gallery_index(folder, detector, embedder)` parcourt les fichiers reconnus
dans la galerie avec une barre de progression. Pour chaque image, le détecteur
retourne tous les visages qui passent ses seuils; chacun est ensuite aligné,
encodé et sauvegardé dans un `FaceRecord`. Avec InsightFace, `FaceAnalysis.get`
a déjà calculé l'embedding ArcFace avec la détection : l'indexeur réutilise son
`normed_embedding` au lieu de lancer une seconde inférence sur le crop. Les
autres implémentations de `FaceEmbedder` reçoivent le crop aligné. Une image peut
ainsi produire zéro, un ou plusieurs enregistrements.

Les images illisibles et celles où aucun visage n'est détecté sont ignorées et
signalées dans le journal. Les erreurs du détecteur, de l'alignement ou du modèle
d'embedding ne sont pas cachées : elles remontent à l'appelant.

Par défaut, l'index est mis en cache dans
`configs/config.yaml` → `paths.cache_dir` → `embeddings.npz` (actuellement
`data/cache/embeddings.npz`). Le fichier NPZ compressé contient les embeddings,
les crops et les métadonnées nécessaires aux `FaceRecord`. Il est écrit de
manière atomique et lu sans activer le pickle.

Avant réutilisation, le code recalcule une empreinte SHA-256 des noms, dates de
modification et octets de toutes les images. Il compare aussi l'empreinte de la configuration
du détecteur, du modèle et du backend d'embedding. Si la galerie ou ces
paramètres ont changé, l'index est reconstruit. Au deuxième passage valide,
les images ne sont pas redétectées et les embeddings ne sont pas recalculés.
Le calcul de l'empreinte relit cependant les fichiers de galerie.

Exemple d'appel en réutilisant une seule instance InsightFace :

```python
from src.detection import FaceDetector
from src.embedding import InsightFaceEmbedder
from src.gallery import build_gallery_index

embedder = InsightFaceEmbedder()
detector = FaceDetector(analysis=embedder.analysis)
records = build_gallery_index("data/gallery", detector, embedder)
print(f"{len(records)} visages indexés")
```

Le chemin du cache peut être remplacé pour un test ou un usage ponctuel :
`build_gallery_index(..., cache_path=Path("/chemin/vers/cache.npz"))`.

### Vérification réelle de l'étape 5

L'indexeur a été exécuté sur les 29 images actuellement dans `data/gallery`.
Le moteur a détecté et indexé 29 visages. Le cache a ensuite été rechargé avec
29 enregistrements en environ `0,08 s`, sans nouvelle inférence ArcFace; la
première indexation a pris environ `59,93 s` sur l'environnement de travail.
Ces durées dépendent de la machine, mais le second passage confirme que les
embeddings ne sont pas recalculés.

## Suite des étapes

1. **Détection** : examiner les boîtes et landmarks sur des images réelles et
   confirmer la qualité de détection.
2. **Alignement** : implémenté dans `FaceDetector.align`; la comparaison visuelle
   des crops dans le notebook reste à faire.
3. **Embeddings** : implémenté pour produire des vecteurs `float32` normalisés L2;
   le critère métier de comparaison de paires réelles reste à vérifier avec des
   identités confirmées.
4. **Galerie et cache** : implémenté et vérifié sur les 29 images; une deuxième
   exécution recharge les 29 enregistrements sans nouvelle inférence.
5. **Recherche** : comparer les embeddings par similarité cosinus et retenir le
   meilleur visage de chaque image.
6. **Identité et pipeline** : élargir éventuellement la recherche, exporter les
   résultats et mesurer la qualité avec les labels.

Les dépendances Python et critères de qualité sont décrits dans
[docs/TECH_STACK.md](./TECH_STACK.md). La progression et les critères « Done »
restent suivis dans [docs/ROADMAP.md](./ROADMAP.md).
