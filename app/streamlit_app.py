"""Interface Streamlit pour rechercher une identité dans une galerie locale."""

from __future__ import annotations

import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st
from PIL import Image, UnidentifiedImageError

# Streamlit starts the script with `app/` on sys.path, not necessarily the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config
from src.io_utils import list_images
from src.pipeline import run

SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
RESULT_STATE_KEY = "face_search_result"
ERROR_STATE_KEY = "face_search_error"


def build_results_archive(output_dir: str | Path) -> bytes:
    """Crée en mémoire un ZIP avec matches, annotations, grille et rapport."""
    root = Path(output_dir)
    archive_buffer = BytesIO()
    files_to_include = [
        path
        for directory in ("matches", "annotated")
        for path in sorted((root / directory).glob("*"))
        if path.is_file()
    ]
    files_to_include.extend(
        path
        for path in (
            root / "result_grid.png",
            root / "similarity_heatmap.png",
            root / "report.json",
        )
        if path.is_file()
    )
    if not files_to_include:
        raise FileNotFoundError(f"Aucun résultat exportable trouvé dans '{root}'.")

    with ZipFile(archive_buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for file_path in files_to_include:
            archive.write(file_path, arcname=file_path.relative_to(root).as_posix())
    return archive_buffer.getvalue()


def _is_valid_image(image_bytes: bytes) -> bool:
    """Vérifie le contenu uploadé avant de l'afficher ou de lancer le pipeline."""
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        return False
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                return False
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
        return False
    return True


def _run_search(
    uploaded_bytes: bytes,
    uploaded_name: str,
    gallery_dir: Path,
    config: AppConfig,
    threshold: float,
    use_cache: bool,
) -> tuple[list[dict[str, str | float | int]], bytes, bytes]:
    """Exécute le pipeline dans un dossier temporaire et prépare les artefacts UI."""
    safe_name = Path(uploaded_name).name
    suffix = Path(safe_name).suffix.lower()
    if suffix.lstrip(".") not in SUPPORTED_UPLOAD_TYPES:
        suffix = ".jpg"
    with TemporaryDirectory(prefix="face-recognition-streamlit-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        query_path = temporary_root / f"query{suffix}"
        output_dir = temporary_root / "output"
        query_path.write_bytes(uploaded_bytes)

        updated_config = replace(
            config,
            matching=replace(config.matching, threshold=threshold),
        )
        matches = run(
            query_path,
            gallery_dir,
            output_dir,
            updated_config,
            use_cache=use_cache,
        )
        archive_bytes = build_results_archive(output_dir)
        grid_bytes = (output_dir / "result_grid.png").read_bytes()
        serialized_matches = [
            {
                "filename": Path(match.image_path).name,
                "image_path": match.image_path,
                "score": match.best_score,
                "matched_face_index": match.matched_face_index,
            }
            for match in matches
        ]
    return serialized_matches, grid_bytes, archive_bytes


def _display_previous_result() -> None:
    """Affiche les résultats conservés après le rerun Streamlit."""
    result = st.session_state.get(RESULT_STATE_KEY)
    if not result:
        return

    st.subheader("Résultats")
    st.image(
        result["grid"], caption="Requête et meilleures correspondances", use_container_width=True
    )
    matches = result["matches"]
    if matches:
        st.success(f"{len(matches)} image(s) correspondante(s) trouvée(s).")
        columns = st.columns(3)
        for index, match in enumerate(matches):
            with columns[index % len(columns)]:
                st.markdown(f"**{match['filename']}**")
                st.caption(
                    f"Similarité cosinus : {match['score']:.3f} · "
                    f"visage : {match['matched_face_index']}"
                )
    else:
        st.info("Aucune correspondance n’a dépassé le seuil sélectionné.")

    st.download_button(
        label="Télécharger les résultats (ZIP)",
        data=result["archive"],
        file_name="face-recognition-results.zip",
        mime="application/zip",
        use_container_width=True,
    )


def main() -> None:
    """Construit l'interface et déclenche la recherche à la demande."""
    st.set_page_config(
        page_title="Recherche faciale locale",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Recherche faciale dans une galerie locale")
    st.write(
        "Importez une image requête, choisissez votre dossier de galerie et lancez "
        "une recherche. Les fichiers restent sur cette machine."
    )

    config = load_config()
    with st.sidebar:
        st.header("Paramètres")
        gallery_value = st.text_input(
            "Dossier de la galerie",
            value=str(Path(__file__).resolve().parents[1] / "data" / "gallery"),
            help="Chemin local du dossier contenant les images à rechercher.",
        )
        threshold = st.slider(
            "Seuil de similarité cosinus",
            min_value=0.0,
            max_value=1.0,
            value=float(config.matching.threshold),
            step=0.01,
            help="Un seuil plus élevé réduit les faux positifs, mais peut manquer des images.",
        )
        use_cache = st.checkbox(
            "Réutiliser le cache des embeddings",
            value=True,
            help="Désactivez cette option pour recalculer les embeddings de la galerie.",
        )

    with st.form("face_search_form"):
        uploaded_file = st.file_uploader(
            "Image requête",
            type=SUPPORTED_UPLOAD_TYPES,
            help="Formats acceptés : JPG, PNG, BMP et WebP.",
        )
        submitted = st.form_submit_button(
            "Rechercher dans la galerie",
            type="primary",
            use_container_width=True,
        )

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        if _is_valid_image(uploaded_bytes):
            st.image(uploaded_bytes, caption="Image requête", width=280)
        elif submitted:
            st.error("Le fichier importé n’est pas une image lisible.")

    if submitted:
        st.session_state.pop(RESULT_STATE_KEY, None)
        st.session_state.pop(ERROR_STATE_KEY, None)
        if uploaded_file is None:
            st.error("Importez une image avant de lancer la recherche.")
        elif not _is_valid_image(uploaded_file.getvalue()):
            st.error("Le fichier importé n’est pas une image lisible.")
        else:
            gallery_dir = Path(gallery_value).expanduser().resolve()
            if not gallery_dir.is_dir():
                st.error(f"Le dossier de galerie n’existe pas : {gallery_dir}")
            else:
                try:
                    gallery_images = list_images(gallery_dir)
                    if not gallery_images:
                        st.error("Le dossier sélectionné ne contient aucune image prise en charge.")
                    else:
                        with st.spinner(
                            f"Recherche dans {len(gallery_images)} image(s)… "
                            "La première exécution peut charger les modèles."
                        ):
                            matches, grid_bytes, archive_bytes = _run_search(
                                uploaded_file.getvalue(),
                                uploaded_file.name,
                                gallery_dir,
                                config,
                                threshold,
                                use_cache,
                            )
                        st.session_state[RESULT_STATE_KEY] = {
                            "matches": matches,
                            "grid": grid_bytes,
                            "archive": archive_bytes,
                            "threshold": threshold,
                        }
                except (OSError, ValueError, RuntimeError, ImportError) as error:
                    st.session_state[ERROR_STATE_KEY] = str(error)

    error_message = st.session_state.get(ERROR_STATE_KEY)
    if error_message:
        st.error(f"La recherche a échoué : {error_message}")
    _display_previous_result()

    with st.expander("À propos du traitement et des données"):
        st.markdown(
            "- Le pipeline détecte et compare les visages avec InsightFace/ArcFace.\n"
            "- L’index de la galerie peut être conservé dans `data/cache/` si le cache est activé.\n"
            "- Les résultats téléchargés incluent les images correspondantes, les annotations, "
            "la grille de résultats et le rapport JSON."
        )


if __name__ == "__main__":
    main()
