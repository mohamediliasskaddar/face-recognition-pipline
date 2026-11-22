# TrombiAI – Roadmap V2 (professionnalisation)

> **Pour Copilot** : la V1 fonctionne (détection → embeddings → clustering → dossiers `person_XX/`). La V2 n'ajoute **aucune nouvelle logique de reconnaissance** : elle améliore l'interface, l'identité visuelle, la documentation visuelle et l'observabilité.
> Règle de travail : **une phase à la fois, dans l'ordre**, ne passer à la suivante que si les critères « Done quand » sont validés.
> Prompt de départ : *« Lis docs/ROADMAP_V2.md et docs/SRC_REFERENCE.md. Nous en sommes à la phase VN. Implémente uniquement cette phase. »*

## Suivi global

- [ ] V0 – Préparation (branche, tag, état des lieux)
- [x] V1 – Découpage de l'interface Streamlit (4 fichiers)
- [ ] V2 – Identité visuelle « TrombiAI » (nom + bannière)
- [ ] V3 – Reset total fiable
- [ ] V4 – Commande de trace du pipeline (sorties de chaque étape)
- [ ] V5 – Page HTML/CSS/JS qui explique le pipeline
- [ ] V6 – Finalisation (docs, tests, démo)

## Pourquoi cet ordre

| Phase | Raison de sa position |
|---|---|
| V1 d'abord | Le reset (V3) et la bannière (V2) ont besoin d'un code UI propre : `ui/state.py` pour l'état, `streamlit_app.py` pour l'en-tête |
| V2 avant V3 | Rapide, visible, et valide la nouvelle structure |
| V3 | S'appuie sur `ui/state.py` créé en V1 |
| V4 avant V5 | La trace produit de vraies images/entrées/sorties que la page HTML peut illustrer |
| V5 | Support de présentation, dépend de `docs/SRC_REFERENCE.md` pour les contenus |

---

## V0 – Préparation

**Tâches**
1. `git tag v1.0 && git checkout -b feature/v2`.
2. Lancer l'app et noter le comportement actuel (reset défaillant, fichier `streamlit_app.py` trop gros).
3. Vérifier que `app/public/banner.png` existe.

**Done quand** : tag `v1.0` créé, branche `feature/v2` active, app lancée sans erreur.

---

## V1 – Découpage de l'interface Streamlit (version simple : 4 fichiers)

**Objectif** : sortir la logique du gros `streamlit_app.py` en **4 fichiers seulement**, faciles à retenir et à demander à Copilot. Pas de sous-dossiers de composants.

**Arborescence cible**

```text
app/
├── streamlit_app.py     # 1. ENTRÉE : constantes de l'app, config page, bannière, en-tête, assemblage
├── styles.css           #    (simple fichier de style, pas du code Python)
├── public/
│   └── banner.png
└── ui/
    ├── __init__.py      #    vide, nécessaire pour `from ui import ...`
    ├── state.py         # 2. ÉTAT : session_state, valeurs par défaut, reset
    ├── components.py    # 3. VUES : sidebar, uploader, boutons, stats, résultats, export ZIP
    └── services.py      # 4. PONT vers src/ : lance le pipeline (aucun import streamlit)
```

**Rôle de chaque fichier**

| Fichier | Contient | Taille cible |
|---|---|---|
| `streamlit_app.py` | `APP_NAME`, `APP_VERSION`, chemins ; `setup_page()` ; `render_header()` (bannière + titre + CSS) ; assemblage des sections dans l'ordre | ≤ 80 lignes |
| `ui/state.py` | `DEFAULTS`, `init_state()`, `has_results()`, `reset_all()` | ≤ 60 lignes |
| `ui/components.py` | `render_sidebar()`, `render_uploader()`, `render_actions()`, `render_stats()`, `render_results()`, `render_downloads()` | ≤ 250 lignes |
| `ui/services.py` | `run_clustering(files, params, workdir)`, `run_search(...)`, `make_zip(...)` : reçoivent des chemins/paramètres, renvoient des objets Python simples | ≤ 100 lignes |

**Règles**
- Pas de logique métier dans `streamlit_app.py` ni dans `components.py` : tout appel au pipeline passe par `services.py`.
- `services.py` n'importe **pas** `streamlit` (testable avec pytest).
- L'état (`st.session_state`) est lu et modifié uniquement via les fonctions de `state.py`.
- Les constantes (nom, version, chemins, valeurs par défaut) sont en haut de `streamlit_app.py` ou dans `DEFAULTS` ; pas de valeurs en dur dans les composants.
- Refactor pur : **aucun changement de comportement** visible dans cette phase.

**Squelette cible de `streamlit_app.py`**

```python
import streamlit as st
from ui import state, components

APP_NAME = "TrombiAI"
APP_VERSION = "2.0"

def setup_page(): ...        # st.set_page_config (complété en V2)
def render_header(): ...     # bannière + titre + CSS (complété en V2)

setup_page()
state.init_state()
render_header()

params = components.render_sidebar()
files = components.render_uploader()
components.render_actions(files, params)

if state.has_results():
    components.render_stats()
    components.render_results()
    components.render_downloads()
```

**Prompt Copilot**
> Refactore app/streamlit_app.py selon la phase V1 de docs/ROADMAP_V2.md : exactement 4 fichiers (streamlit_app.py, ui/state.py, ui/components.py, ui/services.py) plus ui/__init__.py vide. Déplace le code existant sans changer le comportement. services.py ne doit pas importer streamlit. streamlit_app.py doit faire moins de 80 lignes.

**Done quand** : `streamlit run app/streamlit_app.py` fonctionne comme avant, il y a 4 fichiers Python (plus `__init__.py`), et `services.py` ne contient aucun `import streamlit`.

---

## V2 – Identité visuelle « TrombiAI »

**Objectif** : donner une identité à l'application.

**Tâches**
1. Renommer partout le titre en **TrombiAI** (`APP_NAME`), avec un slogan court (ex. *« Trie tes photos par personne, automatiquement »*).
2. `setup_page()` (dans `streamlit_app.py`) : `st.set_page_config(page_title="TrombiAI", page_icon=..., layout="wide")`.
3. `render_header()` (dans `streamlit_app.py`) : afficher `app/public/banner.png` à l'entrée de l'app (pleine largeur), puis le titre et le slogan.
   - Chemin robuste : `Path(__file__).resolve().parent.parent / "public" / "banner.png"` (ne pas dépendre du dossier de lancement).
   - Si le fichier est absent : fallback texte, sans planter.
4. `app/styles.css` : palette cohérente (couleur principale tirée de la bannière), polices, cartes arrondies, boutons ; injecté via `st.markdown("<style>…</style>", unsafe_allow_html=True)`.
5. Pied de page discret : nom, version (`APP_VERSION = "2.0"`), module/projet.

**Prompt Copilot**
> Implémente la phase V2 de docs/ROADMAP_V2.md : renommer l'app en TrombiAI, afficher app/public/banner.png en haut avec un chemin basé sur __file__, charger styles.css, et prévoir un fallback si l'image est absente.

**Done quand** : l'onglet du navigateur affiche « TrombiAI », la bannière apparaît à l'entrée, et l'app ne plante pas si la bannière est supprimée.

---

## V3 – Reset total fiable

**Problème actuel** : le bouton Reset vide les résultats, mais les images sélectionnées dans « Dossier de référence à classer » restent affichées et doivent être retirées une par une.

**Cause** : un `st.file_uploader` ne peut pas être vidé par code. Tant que sa `key` est identique, Streamlit conserve les fichiers sélectionnés.

**Solution à implémenter** : *rotation de la clé* du widget.

```python
# state.py
DEFAULTS = {"uploader_key": 0, "results": None, "run_id": None, "workdir": None}

def init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

def reset_all():
    # 1. supprimer les fichiers temporaires (uploads + sorties) de cette session
    wd = st.session_state.get("workdir")
    if wd:
        shutil.rmtree(wd, ignore_errors=True)
    # 2. réinitialiser tout l'état applicatif
    for k, v in DEFAULTS.items():
        st.session_state[k] = v if k != "uploader_key" else st.session_state["uploader_key"] + 1
    # 3. changer la clé du file_uploader => widget recréé vide
```

```python
# ui/components.py -> render_uploader()
files = st.file_uploader(
    "Dossier de référence à classer",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

# ui/components.py -> render_actions() : utiliser on_click (le callback s'exécute avant le rerun)
st.button("Réinitialiser", on_click=state.reset_all, type="secondary")
```

**Comportement attendu du reset (liste complète)**

| Élément | Après Reset |
|---|---|
| Images sélectionnées dans l'uploader | Toutes retirées d'un clic |
| Résultats (personnes, crops, stats) | Effacés |
| Fichiers temporaires (uploads, sorties de session) | Supprimés du disque |
| Paramètres de la sidebar | Option : conservés (par défaut) ou remis aux valeurs par défaut (bouton séparé) |
| Modèle chargé (InsightFace) | **Conservé** (`st.cache_resource`) pour ne pas le recharger |
| Cache d'embeddings de la galerie | Conservé sauf si `Reset complet` coché |

**Détails**
- Chaque session travaille dans son propre dossier temporaire (`tempfile.mkdtemp(prefix="trombiai_")`) stocké dans `st.session_state.workdir`, pour ne jamais supprimer les données d'un autre utilisateur ni le dossier `data/` du projet.
- Ne jamais supprimer un chemin qui n'est pas sous le dossier temporaire de la session (garde-fou).
- Ajouter une confirmation légère si des résultats existent (`st.toast("Session réinitialisée")` suffit).

**Prompt Copilot**
> Implémente la phase V3 de docs/ROADMAP_V2.md. Le reset doit vider le file_uploader en faisant tourner sa key (uploader_{n}), supprimer le dossier temporaire de session, effacer résultats et état, et conserver le modèle en cache. Utilise on_click sur le bouton. Ajoute un test unitaire de reset_all avec un faux session_state.

**Done quand** (test manuel) :
1. Charger 10 images → lancer le tri → résultats affichés.
2. Cliquer sur Reset → l'uploader est **vide**, aucun résultat, aucun fichier restant dans le dossier temporaire.
3. Charger un autre dossier immédiatement → tout fonctionne comme au premier lancement.

---

## V4 – Commande de trace du pipeline

**Objectif** : une commande qui exécute le pipeline **étape par étape** et sauvegarde l'**entrée et la sortie de chaque étape**. Sert à déboguer, à la démo orale et à alimenter la page HTML (V5).

**Commande**

```bash
# une seule image
python -m src.main --mode trace --input data/gallery/img_03.jpg --output outputs/trace/

# un dossier complet (inclut le clustering)
python -m src.main --mode trace --input data/gallery --output outputs/trace/ --max-images 10
```

**Implémentation**
- Nouveau module `src/trace.py` avec une classe `Tracer` :
  - `Tracer.step(name, inputs_desc, outputs_desc)` : context manager qui mesure la durée et enregistre ce qui entre/sort.
  - `Tracer.save_image(step, name, img)`, `Tracer.save_array(step, name, arr)`, `Tracer.save_json(step, name, obj)`.
  - `Tracer.report()` : affiche un tableau console (rich/tabulate ou simple) et écrit `trace_report.md` + `trace.json`.
- `pipeline.run_cluster(..., tracer: Tracer | None = None)` : si `tracer` est fourni, chaque étape y enregistre ses sorties. Si `None` : aucun surcoût (la prod n'est pas affectée).
- Les noms d'étapes et leurs descriptions viennent de `docs/SRC_REFERENCE.md` (mêmes identifiants `S01`…`S10`).

**Sorties produites**

```text
outputs/trace/
├── S01_ingestion/        # liste des images valides, tailles, images ignorées
├── S02_detection/        # image + boîtes + 5 landmarks dessinés, detections.json
├── S03_qualite/          # scores (confiance, taille, netteté) + visages retenus/écartés
├── S04_alignement/       # visages alignés 112x112 (avant/après côte à côte)
├── S05_embedding/        # embeddings.npy, aperçu (norme, dimension, 8 premières valeurs)
├── S06_similarite/       # heatmap de la matrice de similarité visage x visage
├── S07_clustering/       # labels.json, dendrogramme ou projection 2D (PCA/t-SNE)
├── S08_rattachement/     # visages douteux rattachés / fusions de clusters
├── S09_recadrage/        # crops finaux avec marge
├── S10_organisation/     # arborescence person_XX/ générée
├── trace.json            # durée + entrées/sorties de chaque étape
└── trace_report.md       # rapport lisible
```

**Affichage console attendu**

```text
Étape               Entrée                      Sortie                          Durée
S01 Ingestion       dossier (30 fichiers)       30 images valides               0.2 s
S02 Détection       30 images                   47 visages                      4.1 s
S03 Qualité         47 visages                  41 fiables / 6 douteux          0.3 s
S04 Alignement      47 visages (bbox+points)    47 crops 112x112                0.2 s
S05 Embedding       47 crops                    47 vecteurs 512-d               3.8 s
S07 Clustering      41 vecteurs                 5 clusters                      0.1 s
S08 Rattachement    6 douteux + 5 centroïdes    4 rattachés, 2 unknown          0.0 s
S10 Organisation    47 visages + labels         5 dossiers + unknown/           0.4 s
```

**Prompt Copilot**
> Implémente la phase V4 de docs/ROADMAP_V2.md : crée src/trace.py (classe Tracer), ajoute le paramètre optionnel tracer à pipeline.run_cluster, enregistre les sorties de chaque étape S01..S10 selon docs/SRC_REFERENCE.md, ajoute --mode trace au CLI et génère trace.json + trace_report.md.

**Done quand** : la commande ci-dessus produit tous les dossiers `Sxx_*`, le tableau console, et `trace.json` valide ; le mode normal n'est pas ralenti.

---

## V5 – Page HTML/CSS/JS qui explique le pipeline

**Objectif** : une page statique (sans serveur, ouvrable par double-clic) qui **visualise le pipeline**, avec pour chaque étape son **rôle, son entrée, sa sortie, son module `src/`** et ses paramètres. Utile pour expliquer le projet à l'oral.

**Arborescence**

```text
docs/pipeline/
├── index.html
├── style.css
├── script.js
├── stages.json            # source de vérité : étapes, entrées/sorties (copie de SRC_REFERENCE.md)
└── assets/                # images issues de la trace V4 (optionnel : exemples réels)
```

**Contenu de la page**
1. **En-tête TrombiAI** (logo/bannière réutilisée depuis `app/public/banner.png`) + phrase de présentation.
2. **Schéma de flux horizontal/vertical** : une carte par étape, reliées par des flèches animées (CSS). Deux parcours visibles : **Mode Tri (clustering)** et **Mode Recherche (requête)**, sélectionnables par onglets.
3. **Chaque carte contient** : numéro + nom, module `src/…`, **Entrée** (badge), **Sortie** (badge), rôle en une phrase.
4. **Clic sur une carte** → panneau de détails : explication, paramètres (seuils du `config.yaml`), cas limites, exemple d'image avant/après (si présent dans `assets/`).
5. **Bouton « Lecture animée »** : met en surbrillance les étapes une à une en montrant la donnée qui circule (image → visages → vecteurs → groupes → dossiers).
6. **Légende des types de données** (couleurs) : Image, Visage, Vecteur, Groupe, Dossier.
7. Thème cohérent avec l'app (mêmes couleurs que `styles.css`), responsive, mode sombre en option.

**Contraintes techniques**
- HTML/CSS/JS **vanilla**, aucune dépendance ni framework, aucun appel réseau (fonctionne hors ligne, via `file://`).
- `script.js` charge les données depuis une constante JS `STAGES` (dupliquer `stages.json` dedans pour éviter les restrictions CORS en `file://`) ; garder `stages.json` comme référence.
- Accessible : contrastes suffisants, navigation clavier, `aria-label` sur les cartes.

**Prompt Copilot**
> Implémente la phase V5 de docs/ROADMAP_V2.md. Crée docs/pipeline/index.html, style.css, script.js en vanilla. Utilise les étapes S01..S10 et le mode recherche décrits dans docs/SRC_REFERENCE.md. Chaque étape est une carte avec module, entrée, sortie et rôle ; un clic ouvre un panneau de détails ; un bouton lance une animation séquentielle du flux. Deux onglets : mode Tri et mode Recherche.

**Done quand** : la page s'ouvre en double-clic, les deux parcours sont visibles, chaque étape affiche son entrée et sa sortie, et l'animation fonctionne.

---

## V6 – Finalisation

**Tâches**
1. Mettre à jour `README.md` : nom TrombiAI, capture de l'app, lien vers `docs/pipeline/index.html`, commandes (`streamlit run`, `--mode trace`).
2. Mettre à jour `.github/copilot-instructions.md` (bloc ci-dessous).
3. Tests : `pytest -q` vert (reset, services, trace). Ajouter un test que `services.py` n'importe pas `streamlit`.
4. `ruff check` + `black`.
5. Parcours de démo (5 minutes) : ouvrir la page pipeline → expliquer chaque étape → lancer `--mode trace` sur 1 image → lancer l'app → tri d'un dossier → reset → nouveau dossier.
6. Tag `v2.0`.

**Bloc à ajouter dans `.github/copilot-instructions.md`**

```markdown
## Version 2 (professionnalisation) – TrombiAI
- Nom de l'application : TrombiAI. Suivre docs/ROADMAP_V2.md phase par phase (V0 → V6).
- L'UI tient en 4 fichiers : app/streamlit_app.py (entrée + en-tête), app/ui/state.py, app/ui/components.py, app/ui/services.py. Pas de logique métier dans l'UI.
- app/ui/services.py est le seul pont vers src/ et n'importe jamais streamlit. Ne pas créer d'autres fichiers UI sans demande.
- Le reset utilise la rotation de la key du file_uploader (uploader_{n}) et supprime le dossier temporaire de session ; ne jamais supprimer de chemin hors de ce dossier.
- Le rôle et les entrées/sorties de chaque module src/ sont décrits dans docs/SRC_REFERENCE.md (identifiants S01..S10) : le mode --mode trace et la page docs/pipeline/ doivent rester cohérents avec ce fichier.
```

**Done quand** : tous les éléments de la liste de suivi sont cochés et la démo se déroule sans erreur.
