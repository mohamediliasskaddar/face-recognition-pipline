# Guide de test R8

## Tests automatisés

Depuis la racine du dépôt, activez l'environnement du projet puis exécutez :

```bash
venv/bin/python -m pytest -q
venv/bin/ruff check src app tests
venv/bin/black --check src app tests
```

Les tests unitaires couvrent notamment les dossiers sans image ou sans visage,
un visage unique, plusieurs visages issus de la même image et deux embeddings
très proches. Les tests d'intégration InsightFace utilisent les images locales
de `tests/fixtures/`; ils sont ignorés si le modèle `buffalo_l` n'est pas déjà
installé.

## Tableau de bord Streamlit

Lancez l'application localement :

```bash
streamlit run app/streamlit_app.py
```

1. Dans **Mode de traitement**, choisissez **Tri par personne**.
2. Dans **Dossier d’images à classer**, sélectionnez un dossier. Les sous-dossiers
   sont inclus; les formats pris en charge sont JPG, JPEG, PNG, BMP et WebP.
   Limites de traitement : 20 Mio par image et 250 Mio au total.
3. Réglez la distance cosinus, la taille minimale et la marge des crops.
4. Cliquez sur **Détecter et regrouper les visages**.
5. Vérifiez le nombre de visages, de personnes et d’inconnus; ouvrez les dossiers
   `person_XX` pour examiner leurs planches de contact.
6. Téléchargez l’archive ZIP pour vérifier les crops, les planches et `report.json`.

Les images importées sont traitées localement dans un répertoire temporaire; le
cache est désactivé pour ce flux. Streamlit 1.49 ou une version ultérieure est
nécessaire pour l’import direct d’un dossier.

## Scénarios manuels

- **Aucun visage** : importez un dossier ne contenant que
  `tests/fixtures/pic-with-no-face.png`. Le tableau de bord doit afficher zéro
  visage sans échouer.
- **Plusieurs visages dans une image** : importez un dossier contenant
  `tests/fixtures/three-image-fusionne.jpg`. Le rapport doit compter chaque
  visage détecté séparément et l’archive doit contenir plusieurs crops.
- **Un seul visage** : importez une image nette contenant une seule personne;
  vérifiez qu’un crop et un dossier personne ou `unknown/` sont créés.
- **Identités proches** : dans les tests automatisés, deux embeddings proches
  forment deux groupes avec un seuil faible et un groupe avec un seuil plus
  élevé. Dans l’interface, comparez les résultats en déplaçant **Distance
  cosinus**.
- **Recherche historique** : choisissez **Recherche historique**, importez une
  image requête et vérifiez que la galerie locale et les résultats de recherche
  restent accessibles.
