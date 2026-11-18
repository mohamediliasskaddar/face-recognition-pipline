"""Tableau de bord Streamlit pour le clustering et la recherche faciale."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path, PurePosixPath
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
from src.pipeline import run, run_cluster

SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_CLUSTER_UPLOAD_FILES = 500
MAX_CLUSTER_UPLOAD_BYTES = 250 * 1024 * 1024
RESULT_STATE_KEY = "face_search_result"
ERROR_STATE_KEY = "face_search_error"
CLUSTER_RESULT_STATE_KEY = "face_cluster_result"
CLUSTER_ERROR_STATE_KEY = "face_cluster_error"


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


def build_cluster_archive(output_dir: str | Path) -> bytes:
    """Crée un ZIP des crops, planches de contact et du rapport de clustering."""
    root = Path(output_dir)
    files_to_include = [
        path
        for directory in ("person_*", "unknown", "contact_sheets")
        for folder in sorted(root.glob(directory))
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    ]
    report_path = root / "report.json"
    if report_path.is_file():
        files_to_include.append(report_path)
    if not files_to_include:
        raise FileNotFoundError(f"Aucun résultat de clustering exportable dans '{root}'.")

    archive_buffer = BytesIO()
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


def _write_uploaded_gallery(
    uploaded_images: Sequence[tuple[str, bytes]],
    gallery_dir: str | Path,
) -> int:
    """Valide et écrit les images importées dans une galerie temporaire."""
    if not uploaded_images:
        raise ValueError("Importez un dossier contenant au moins une image.")
    if len(uploaded_images) > MAX_CLUSTER_UPLOAD_FILES:
        raise ValueError(f"Le dossier dépasse la limite de {MAX_CLUSTER_UPLOAD_FILES} fichiers.")
    total_bytes = sum(len(image_bytes) for _, image_bytes in uploaded_images)
    if total_bytes > MAX_CLUSTER_UPLOAD_BYTES:
        raise ValueError("La taille totale des images dépasse la limite de 250 Mio.")

    destination_root = Path(gallery_dir).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    written_paths: set[Path] = set()
    for uploaded_name, image_bytes in uploaded_images:
        normalized_name = uploaded_name.replace("\\", "/")
        relative_path = PurePosixPath(normalized_name)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"..", "."} for part in relative_path.parts)
            or relative_path.parts[0].endswith(":")
        ):
            raise ValueError(f"Chemin de fichier importé invalide : '{uploaded_name}'.")
        if relative_path.suffix.lower().lstrip(".") not in SUPPORTED_UPLOAD_TYPES:
            raise ValueError(f"Format d'image non pris en charge : '{uploaded_name}'.")
        if not _is_valid_image(image_bytes):
            raise ValueError(f"Image vide, illisible ou trop volumineuse : '{uploaded_name}'.")

        destination = destination_root.joinpath(*relative_path.parts).resolve()
        if not destination.is_relative_to(destination_root):
            raise ValueError(f"Chemin de fichier importé invalide : '{uploaded_name}'.")
        if destination in written_paths or destination.exists():
            raise ValueError(f"Deux fichiers importés ont le même chemin : '{uploaded_name}'.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        written_paths.add(destination)
    return len(written_paths)


def _run_cluster(
    uploaded_images: Sequence[tuple[str, bytes]],
    config: AppConfig,
    distance_threshold: float,
    min_face_size: int,
    margin: float,
) -> tuple[dict[str, object], dict[str, bytes], bytes]:
    """Exécute le clustering des images importées et conserve les artefacts en mémoire."""
    if not math.isfinite(distance_threshold) or not 0.0 <= distance_threshold <= 2.0:
        raise ValueError("Le seuil de distance doit être compris entre 0 et 2.")
    if min_face_size <= 0:
        raise ValueError("La taille minimale du visage doit être positive.")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("La marge doit être un nombre fini positif ou nul.")

    with TemporaryDirectory(prefix="face-cluster-streamlit-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        gallery_dir = temporary_root / "gallery"
        output_dir = temporary_root / "output"
        _write_uploaded_gallery(uploaded_images, gallery_dir)

        updated_config = replace(
            config,
            detection=replace(config.detection, min_face_size=min_face_size),
            quality=replace(config.quality, min_face_size=min_face_size),
            clustering=replace(
                config.clustering,
                distance_threshold=distance_threshold,
            ),
            output=replace(
                config.output,
                crop_margin=margin,
                copy_originals=False,
                overwrite=True,
            ),
        )
        run_cluster(gallery_dir, output_dir, updated_config, use_cache=False)

        report_path = output_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or not isinstance(report.get("folders"), dict):
            raise TypeError("Le pipeline a généré un rapport de clustering invalide.")
        for folder_data in report["folders"].values():
            if not isinstance(folder_data, dict) or not isinstance(folder_data.get("crops"), list):
                raise TypeError("Le rapport de clustering contient des crops invalides.")
            for crop in folder_data["crops"]:
                if not isinstance(crop, dict) or not isinstance(crop.get("source_image"), str):
                    raise TypeError("Le rapport de clustering contient une source invalide.")
                source_path = Path(crop["source_image"]).resolve()
                crop["source_image"] = source_path.relative_to(gallery_dir).as_posix()
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        contact_sheets = {
            path.stem: path.read_bytes()
            for path in sorted((output_dir / "contact_sheets").glob("*.png"))
        }
        archive_bytes = build_cluster_archive(output_dir)
    return report, contact_sheets, archive_bytes


def _display_cluster_result() -> None:
    """Affiche les personnes, inconnus, planches et archive du clustering."""
    result = st.session_state.get(CLUSTER_RESULT_STATE_KEY)
    if not result:
        return

    report = result["report"]
    folders = report["folders"]
    metrics = st.columns(3)
    metrics[0].metric("Visages détectés", report["n_faces"])
    metrics[1].metric("Personnes regroupées", report["n_persons"])
    metrics[2].metric("Visages inconnus", folders["unknown"]["face_count"])

    contact_sheets = result["contact_sheets"]
    if not report["n_faces"]:
        st.info("Aucun visage exploitable n’a été détecté dans le dossier.")
    for folder_name, folder_data in folders.items():
        if not folder_data["crops"]:
            continue
        title = f"{folder_name} · {folder_data['face_count']} visage(s)"
        with st.expander(title, expanded=folder_name.startswith("person_")):
            sheet = contact_sheets.get(folder_name)
            if sheet is not None:
                st.image(sheet, width="stretch")
            st.dataframe(
                [
                    {
                        "Image source": crop["source_image"],
                        "Index du visage": crop["face_index"],
                        "Crop": crop["filename"],
                    }
                    for crop in folder_data["crops"]
                ],
                hide_index=True,
                width="stretch",
            )

    st.download_button(
        label="Télécharger les crops et rapports (ZIP)",
        data=result["archive"],
        file_name="face-clustering-results.zip",
        mime="application/zip",
        width="stretch",
    )


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
    st.image(result["grid"], caption="Requête et meilleures correspondances", width="stretch")
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
        width="stretch",
    )


def _render_cluster_dashboard(config: AppConfig) -> None:
    """Affiche le formulaire d'import de dossier et les résultats de clustering."""
    st.header("Tri des visages par personne")
    st.write(
        "Importez un dossier d’images. Les visages détectés seront regroupés "
        "et les crops resteront disponibles dans cette session."
    )
    with st.sidebar:
        st.subheader("Paramètres du clustering")
        distance_threshold = st.slider(
            "Distance cosinus",
            min_value=0.0,
            max_value=2.0,
            value=float(config.clustering.distance_threshold),
            step=0.01,
            help="Une distance plus faible sépare davantage les groupes.",
        )
        min_face_size = st.slider(
            "Taille minimale du visage (pixels)",
            min_value=10,
            max_value=200,
            value=int(config.quality.min_face_size),
            step=5,
        )
        margin = st.slider(
            "Marge autour des crops",
            min_value=0.0,
            max_value=0.8,
            value=float(config.output.crop_margin),
            step=0.05,
        )

    uploaded_files = st.file_uploader(
        "Dossier d’images à classer",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files="directory",
        help=(
            "Choisissez un dossier contenant des images JPG, PNG, BMP ou WebP. "
            "Les sous-dossiers sont conservés."
        ),
    )
    st.caption("Limites de traitement : 20 Mio par image et 250 Mio pour le dossier.")
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} fichier(s) sélectionné(s).")
    if st.button("Détecter et regrouper les visages", type="primary", width="stretch"):
        st.session_state.pop(CLUSTER_RESULT_STATE_KEY, None)
        st.session_state.pop(CLUSTER_ERROR_STATE_KEY, None)
        if not uploaded_files:
            st.error("Importez un dossier contenant au moins une image.")
        else:
            images = [
                (uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files
            ]
            try:
                with st.spinner(
                    f"Analyse de {len(images)} image(s)… "
                    "La première exécution peut charger les modèles."
                ):
                    report, contact_sheets, archive = _run_cluster(
                        images,
                        config,
                        distance_threshold,
                        min_face_size,
                        margin,
                    )
                st.session_state[CLUSTER_RESULT_STATE_KEY] = {
                    "report": report,
                    "contact_sheets": contact_sheets,
                    "archive": archive,
                }
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                st.session_state[CLUSTER_ERROR_STATE_KEY] = str(error)

    error_message = st.session_state.get(CLUSTER_ERROR_STATE_KEY)
    if error_message:
        st.error(f"Le clustering a échoué : {error_message}")
    _display_cluster_result()

    with st.expander("À propos du traitement et des données"):
        st.markdown(
            "- Le clustering fonctionne sur chaque visage, pas sur l’image entière.\n"
            "- Les crops, les planches de contact et le rapport peuvent être téléchargés en ZIP.\n"
            "- Les fichiers importés sont traités localement et le cache est désactivé pour "
            "ces imports temporaires."
        )


def _render_search_dashboard(config: AppConfig) -> None:
    """Affiche le flux historique de recherche par image requête."""
    st.header("Recherche faciale dans une galerie locale")
    st.write(
        "Importez une image requête, choisissez votre dossier de galerie et lancez "
        "une recherche. Les fichiers restent sur cette machine."
    )
    with st.sidebar:
        st.subheader("Paramètres de recherche")
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
            width="stretch",
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


def main() -> None:
    """Construit le tableau de bord des modes cluster et search."""
    st.set_page_config(
        page_title="Tri et recherche faciale",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Tri et recherche de visages")
    config = load_config()
    with st.sidebar:
        mode = st.radio(
            "Mode de traitement",
            ("Tri par personne", "Recherche historique"),
            key="face_pipeline_mode",
        )
    if mode == "Tri par personne":
        _render_cluster_dashboard(config)
    else:
        _render_search_dashboard(config)


if __name__ == "__main__":
    main()
