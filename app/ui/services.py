"""Services sans dépendance Streamlit pour les traitements du tableau de bord."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, UnidentifiedImageError

from src.config import AppConfig
from src.config import load_config as _load_config
from src.evaluation import load_face_labels, load_labels
from src.io_utils import list_images as _list_images
from src.pipeline import run, run_cluster

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_CLUSTER_UPLOAD_FILES = 500
MAX_CLUSTER_UPLOAD_BYTES = 250 * 1024 * 1024
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "labels.csv"


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Charge la configuration du pipeline."""
    return _load_config(config_path)


def list_images(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    """Liste les images prises en charge dans un dossier."""
    return _list_images(folder, recursive=recursive)


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


def is_valid_image(image_bytes: bytes) -> bool:
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


def resolve_labels_csv(
    uploaded_file: object | None,
    destination_dir: str | Path,
) -> tuple[Path, str]:
    """Prépare le fichier de labels uploadé ou retourne le fichier par défaut."""
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if uploaded_file is not None:
        name = getattr(uploaded_file, "name", "labels.csv")
        safe_name = Path(str(name)).name
        if Path(safe_name).suffix.lower() != ".csv":
            raise ValueError("Le fichier de labels doit être au format CSV.")
        getvalue = getattr(uploaded_file, "getvalue", None)
        if not callable(getvalue):
            raise ValueError("Le fichier de labels uploadé est invalide.")
        contents = getvalue()
        if not isinstance(contents, bytes) or not contents:
            raise ValueError("Le fichier de labels est vide.")
        labels_path = destination / "uploaded_labels.csv"
        labels_path.write_bytes(contents)
        source_label = f"upload: {safe_name}"
    else:
        labels_path = DEFAULT_LABELS_PATH
        source_label = f"par défaut: {labels_path.relative_to(PROJECT_ROOT).as_posix()}"

    if not labels_path.is_file():
        raise FileNotFoundError(f"Aucun fichier de labels disponible : '{labels_path}'.")

    try:
        load_face_labels(labels_path)
    except ValueError:
        load_labels(labels_path)
    return labels_path, source_label


def write_uploaded_gallery(
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
        if not is_valid_image(image_bytes):
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


def run_cluster_uploads(
    uploaded_images: Sequence[tuple[str, bytes]],
    config: AppConfig,
    distance_threshold: float,
    min_face_size: int,
    margin: float,
    workdir: str | Path,
) -> tuple[dict[str, object], dict[str, bytes], bytes]:
    """Exécute le clustering des images importées et conserve les artefacts en mémoire."""
    if not math.isfinite(distance_threshold) or not 0.0 <= distance_threshold <= 2.0:
        raise ValueError("Le seuil de distance doit être compris entre 0 et 2.")
    if min_face_size <= 0:
        raise ValueError("La taille minimale du visage doit être positive.")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("La marge doit être un nombre fini positif ou nul.")

    temporary_root = Path(workdir)
    temporary_root.mkdir(parents=True, exist_ok=True)
    gallery_dir = temporary_root / "gallery"
    output_dir = temporary_root / "output"
    write_uploaded_gallery(uploaded_images, gallery_dir)

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
        )
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


def run_search(
    uploaded_bytes: bytes,
    uploaded_name: str,
    gallery_dir: Path,
    config: AppConfig,
    threshold: float,
    use_cache: bool,
    workdir: str | Path,
) -> tuple[list[dict[str, str | float | int]], bytes, bytes]:
    """Exécute le pipeline dans un dossier temporaire et prépare les artefacts UI."""
    safe_name = Path(uploaded_name).name
    suffix = Path(safe_name).suffix.lower()
    if suffix.lstrip(".") not in SUPPORTED_UPLOAD_TYPES:
        suffix = ".jpg"
    temporary_root = Path(workdir)
    temporary_root.mkdir(parents=True, exist_ok=True)
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
