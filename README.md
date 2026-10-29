# Pipeline de reconnaissance faciale

Pipeline de détection et de reconnaissance faciale multi-instances pour rechercher
les images d'une galerie locale contenant la personne d'une image requête.

Voir [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/TECH_STACK.md](docs/TECH_STACK.md)
et [docs/ROADMAP.md](docs/ROADMAP.md) pour les détails du projet. Le
[guide de suivi](docs/PROJECT_GUIDE.md) résume les composants terminés et les
vérifications qui restent à faire.

## Interface Streamlit

Depuis la racine du dépôt et après installation de `requirements.txt`, lancez :

```bash
streamlit run app/streamlit_app.py
```

L'interface permet d'importer une image requête, de sélectionner un dossier de
galerie local, d'ajuster le seuil cosinus et de télécharger les résultats en ZIP.
Les recherches sont exécutées localement; le cache des embeddings de galerie
peut être activé ou désactivé dans les paramètres.
