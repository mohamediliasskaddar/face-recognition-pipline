# Pipeline de reconnaissance faciale

Pipeline de détection et de reconnaissance faciale multi-instances. Le mode
`cluster` regroupe les visages d'un dossier par personne; le mode historique
`search` recherche dans une galerie les images correspondant à une requête.

Voir [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/TECH_STACK.md](docs/TECH_STACK.md)
et [docs/ROADMAP.md](docs/ROADMAP.md) pour les détails du projet. Le
[guide de suivi](docs/PROJECT_GUIDE.md) résume les composants terminés et les
vérifications qui restent à faire.

## Trier un dossier par personne

```bash
python -m src.main --mode cluster --input data/gallery --output outputs/people
```

Le résultat contient les crops sous `person_XX/` et `unknown/`, les planches de
contact sous `contact_sheets/` et le détail des visages dans `report.json`.
Le seuil `--threshold` est une distance cosinus (de `0` à `2`) dans ce mode.
Les options `--min-face-size`, `--margin`, `--copy-originals` et `--overwrite`
permettent de régler la détection et l'export.

L'évaluation du clustering utilise un CSV par visage avec les colonnes
`filename,face_index,person_id`. Le format historique `filename,person_id` est
également lu et interprété comme un seul visage (index `0`) par image.
`src.evaluation` fournit le calcul des métriques et le balayage des seuils.

## Mode recherche historique

```bash
python -m src.main --mode search --query data/query/person.jpg \
  --gallery data/gallery --output outputs/search
```

Dans ce mode, `--threshold` reste une similarité cosinus (de `-1` à `1`).

## Interface Streamlit

Depuis la racine du dépôt et après installation de `requirements.txt`, lancez :

```bash
streamlit run app/streamlit_app.py
```

L'interface permet d'importer une image requête ou un dossier de référence,
d'importer un dossier à trier par personne, d'ajuster les seuils et de
télécharger les résultats en ZIP. Un fichier de labels CSV peut être importé
dans la barre latérale; à défaut, `data/labels.csv` est utilisé.
Le bouton « Réinitialiser l’interface » efface les résultats et sélections pour
changer rapidement de test. Les recherches sont exécutées localement; le cache
des embeddings de galerie peut être activé ou désactivé dans les paramètres.
