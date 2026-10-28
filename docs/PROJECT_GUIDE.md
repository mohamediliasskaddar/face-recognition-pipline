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
| `src/types.py` | À compléter | Contiendra les structures partagées, dont `FaceRecord` et `MatchResult`. |
| `src/embedding.py` | À compléter | Transformera un visage aligné en vecteur numérique. |
| `src/gallery.py` | À compléter | Indexera les visages de toutes les images et réutilisera un cache. |
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

### Tests sans téléchargement des modèles

Les tests unitaires remplacent le moteur InsightFace par un faux détecteur. Ils
vérifient la configuration transmise, les filtres, l'ordre par aire, la forme
du crop aligné, les entrées invalides et la sélection du visage principal sans
télécharger les poids.
Cela valide la logique Python, mais ne prouve pas encore que les modèles
détectent correctement les visages de la galerie. La validation visuelle avec le
vrai modèle reste nécessaire pour le critère « Done quand » de l'étape 2.

## Suite des étapes

1. **Détection** : examiner les boîtes et landmarks sur des images réelles et
   confirmer la qualité de détection.
2. **Alignement** : implémenté dans `FaceDetector.align`; la comparaison visuelle
   des crops dans le notebook reste à faire.
3. **Embeddings** : produire des vecteurs `float32` normalisés L2.
4. **Galerie et cache** : calculer les embeddings de galerie une fois, puis les
   réutiliser.
5. **Recherche** : comparer les embeddings par similarité cosinus et retenir le
   meilleur visage de chaque image.
6. **Identité et pipeline** : élargir éventuellement la recherche, exporter les
   résultats et mesurer la qualité avec les labels.

Les dépendances Python et critères de qualité sont décrits dans
[docs/TECH_STACK.md](./TECH_STACK.md). La progression et les critères « Done »
restent suivis dans [docs/ROADMAP.md](./ROADMAP.md).
