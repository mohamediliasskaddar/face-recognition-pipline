"""Export des visages recadrés dans des dossiers organisés par personne."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.cropping import crop_face
from src.io_utils import read_image, save_image
from src.types import FaceRecord


class CropEntry(TypedDict):
    """Source et destination d'un crop exporté."""

    filename: str
    source_image: str
    face_index: int


class FolderReport(TypedDict):
    """Résumé des fichiers exportés dans un dossier de personne."""

    label: int
    face_count: int
    crops: list[CropEntry]
    originals: list[str]


class OrganizerReport(TypedDict):
    """Format du rapport JSON d'organisation."""

    n_persons: int
    n_faces: int
    folders: dict[str, FolderReport]


def _validate_labels(
    labels: NDArray[np.generic] | Sequence[int],
    record_count: int,
) -> NDArray[np.int32]:
    """Valide et convertit les labels de clustering."""
    label_array = np.asarray(labels)
    if label_array.shape != (record_count,):
        raise ValueError("labels doit contenir exactement un label par visage.")
    if record_count == 0:
        return np.empty(0, dtype=np.int32)
    if label_array.dtype.kind not in {"i", "u"}:
        raise ValueError("Les labels doivent être des entiers.")
    if label_array.dtype.kind == "u" and np.any(label_array > np.iinfo(np.int32).max):
        raise ValueError("Les labels dépassent la plage des entiers 32 bits.")

    normalized_labels = np.asarray(label_array, dtype=np.int32)
    if np.any(normalized_labels < -1):
        raise ValueError("Le label -1 est la seule valeur négative autorisée.")
    return normalized_labels


def _resolve_output_directory(
    output_dir: str | Path,
    records: Sequence[FaceRecord],
) -> Path:
    """Résout le dossier de sortie et refuse les cibles dangereuses."""
    requested_path = Path(output_dir).expanduser()
    if requested_path.is_symlink():
        raise ValueError("Le dossier de sortie ne peut pas être un lien symbolique.")

    destination = requested_path.resolve()
    if destination == Path(destination.anchor):
        raise ValueError("La racine du système ne peut pas être utilisée comme dossier de sortie.")
    module_path = Path(__file__).resolve()
    if module_path.is_relative_to(destination):
        raise ValueError("Le dossier de sortie ne peut pas contenir le code du projet.")

    for record in records:
        source_path = Path(record.image_path).expanduser().resolve()
        if source_path.is_relative_to(destination) or destination.is_relative_to(
            source_path.parent
        ):
            raise ValueError(
                "Le dossier de sortie ne peut pas contenir ou remplacer les images source."
            )
    return destination


def _build_export_plan(
    records: Sequence[FaceRecord],
    labels: NDArray[np.int32],
    folder_by_label: dict[int, str],
    copy_originals: bool,
) -> list[tuple[FaceRecord, int, str, str]]:
    """Prépare les chemins de crops et détecte les collisions avant l'écriture."""
    face_indices: defaultdict[Path, int] = defaultdict(int)
    used_names: dict[str, set[str]] = defaultdict(set)
    original_paths: dict[str, dict[str, Path]] = defaultdict(dict)
    export_plan: list[tuple[FaceRecord, int, str, str]] = []

    for record, label in zip(records, labels, strict=True):
        source_path = Path(record.image_path)
        resolved_source = source_path.resolve()
        face_index = face_indices[resolved_source]
        face_indices[resolved_source] += 1
        folder_name = folder_by_label.get(int(label), "unknown")
        crop_name = f"{source_path.stem}_face{face_index}.jpg"
        if crop_name in used_names[folder_name]:
            raise ValueError(
                f"Plusieurs crops produisent le même nom dans '{folder_name}': '{crop_name}'."
            )
        used_names[folder_name].add(crop_name)
        export_plan.append((record, face_index, folder_name, crop_name))

        if copy_originals and folder_name != "unknown":
            previous_source = original_paths[folder_name].get(source_path.name)
            if previous_source is not None and previous_source != resolved_source:
                raise ValueError(
                    f"Plusieurs images source nommées '{source_path.name}' "
                    f"seraient copiées dans '{folder_name}'."
                )
            if previous_source is None and source_path.name in used_names[folder_name]:
                raise ValueError(
                    f"Le nom de l'image source '{source_path.name}' entre en collision "
                    f"dans '{folder_name}'."
                )
            if previous_source is None:
                original_paths[folder_name][source_path.name] = resolved_source
                used_names[folder_name].add(source_path.name)

    return export_plan


def organize(
    records: Sequence[FaceRecord],
    labels: NDArray[np.generic] | Sequence[int],
    output_dir: str | Path,
    copy_originals: bool = False,
    *,
    margin: float = 0.25,
    overwrite: bool = False,
) -> Path:
    """Exporte les crops de visage classés dans des dossiers par personne.

    Les personnes sont numérotées par effectif décroissant ; en cas d'égalité,
    le plus petit label d'origine vient en premier. Le label `-1` est exporté
    dans `unknown/`. Un dossier de sortie non vide est refusé, sauf si
    `overwrite=True`.

    Args:
        records: Un enregistrement par visage à exporter.
        labels: Un label par enregistrement, avec `-1` pour les inconnus.
        output_dir: Dossier racine de l'arborescence générée.
        copy_originals: Copie aussi chaque image source dans son dossier personne.
        margin: Marge passée à `crop_face`.
        overwrite: Remplace le contenu existant du dossier de sortie si activé.

    Returns:
        Chemin du fichier `report.json` généré.

    Raises:
        FileExistsError: Si le dossier de sortie contient déjà des fichiers et
            que `overwrite` est désactivé.
        ValueError: Si les labels, les chemins ou les noms de sortie sont invalides.
        OSError: Si une image ne peut pas être lue, écrite ou copiée.
    """
    normalized_labels = _validate_labels(labels, len(records))
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin doit être un nombre fini positif ou nul.")
    destination = _resolve_output_directory(output_dir, records)

    cluster_sizes = {
        int(label): int(np.count_nonzero(normalized_labels == label))
        for label in np.unique(normalized_labels)
        if label >= 0
    }
    ordered_labels = sorted(cluster_sizes, key=lambda label: (-cluster_sizes[label], label))
    folder_by_label = {
        label: f"person_{index:02d}" for index, label in enumerate(ordered_labels, start=1)
    }
    export_plan = _build_export_plan(
        records,
        normalized_labels,
        folder_by_label,
        copy_originals,
    )

    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(f"Le chemin de sortie n'est pas un dossier : '{destination}'.")
        if any(destination.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Le dossier de sortie n'est pas vide : '{destination}'. "
                "Activez overwrite pour le remplacer."
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as temporary_directory:
        staging_directory = Path(temporary_directory) / "output"
        staging_directory.mkdir()
        for folder_name in [*folder_by_label.values(), "unknown"]:
            (staging_directory / folder_name).mkdir()

        folders_report: dict[str, FolderReport] = {
            folder_name: {
                "label": label,
                "face_count": cluster_sizes[label],
                "crops": [],
                "originals": [],
            }
            for label, folder_name in folder_by_label.items()
        }
        folders_report["unknown"] = {
            "label": -1,
            "face_count": int(np.count_nonzero(normalized_labels == -1)),
            "crops": [],
            "originals": [],
        }

        image_cache: dict[Path, NDArray[np.uint8]] = {}
        copied_originals: dict[str, set[Path]] = defaultdict(set)
        for record, face_index, folder_name, crop_name in export_plan:
            source_path = Path(record.image_path)
            resolved_source = source_path.resolve()
            if resolved_source not in image_cache:
                image_cache[resolved_source] = read_image(source_path)
            crop = crop_face(image_cache[resolved_source], record.bbox, margin=margin)
            save_image(staging_directory / folder_name / crop_name, crop)
            folders_report[folder_name]["crops"].append(
                {
                    "filename": crop_name,
                    "source_image": record.image_path,
                    "face_index": face_index,
                }
            )

            if (
                copy_originals
                and folder_name != "unknown"
                and resolved_source not in copied_originals[folder_name]
            ):
                shutil.copy2(source_path, staging_directory / folder_name / source_path.name)
                folders_report[folder_name]["originals"].append(source_path.name)
                copied_originals[folder_name].add(resolved_source)

        report: OrganizerReport = {
            "n_persons": len(ordered_labels),
            "n_faces": len(records),
            "folders": folders_report,
        }
        report_path = staging_directory / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging_directory, destination)

    return destination / "report.json"
