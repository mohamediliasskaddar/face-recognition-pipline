"""Visualisation des détections, résultats et similarités."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.io_utils import read_image
from src.matching import cosine_similarity_matrix
from src.types import FaceRecord, MatchResult


def draw_faces(
    img: NDArray[np.uint8],
    records: list[FaceRecord],
) -> NDArray[np.uint8]:
    """Dessine les boîtes de détection et scores sur une image BGR.

    Args:
        img: Image couleur BGR de type uint8.
        records: Visages à annoter dans cette image.

    Returns:
        Copie annotée de l'image; l'image d'entrée n'est pas modifiée.

    Raises:
        TypeError: Si l'image n'est pas un tableau BGR uint8.
        ValueError: Si l'image est vide ou si une boîte/un score est invalide.
    """
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise TypeError("L'image doit être un tableau numpy BGR de type uint8.")
    if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("L'image BGR doit avoir la forme (H, W, 3) et ne pas être vide.")

    annotated = img.copy()
    for record in records:
        bbox = np.asarray(record.bbox, dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError(f"Boîte invalide pour le visage de '{record.image_path}'.")
        if not np.isfinite(record.det_score):
            raise ValueError(f"Score de détection invalide pour '{record.image_path}'.")

        coordinates = np.rint(bbox).astype(np.int32).tolist()
        x1, y1, x2, y2 = coordinates
        color = (0, 220, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{record.det_score:.2f}"
        text_y = max(y1 - 8, 14)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def make_result_grid(
    query: NDArray[np.uint8],
    matches: list[MatchResult],
    records: list[FaceRecord],
    output_path: str | Path,
) -> None:
    """Crée et enregistre une grille de la requête et des résultats annotés.

    Args:
        query: Image BGR de la requête, annotée si les repères doivent apparaître.
        matches: Résultats classés par score décroissant.
        records: Index des visages, utilisé pour annoter les résultats.
        output_path: Fichier image de sortie, généralement `result_grid.png`.

    Raises:
        OSError: Si une image de résultat ne peut pas être lue ou le fichier
            de grille ne peut pas être écrit.
        ValueError: Si les dimensions de l'image requête sont invalides.
    """
    if (
        not isinstance(query, np.ndarray)
        or query.dtype != np.uint8
        or query.ndim != 3
        or query.shape[2] != 3
        or query.shape[0] == 0
        or query.shape[1] == 0
    ):
        raise ValueError("L'image requête doit être BGR uint8 de forme (H, W, 3).")

    records_by_image: dict[str, list[FaceRecord]] = defaultdict(list)
    for record in records:
        records_by_image[record.image_path].append(record)

    image_count = len(matches) + 1
    columns = min(3, image_count)
    rows = (image_count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    flat_axes[0].imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
    flat_axes[0].set_title("Requête", fontsize=11)
    flat_axes[0].axis("off")

    for axis, match in zip(flat_axes[1:], matches, strict=False):
        image = read_image(match.image_path)
        annotated = draw_faces(image, records_by_image.get(match.image_path, []))
        axis.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f"Cosinus : {match.best_score:.3f}\n{Path(match.image_path).name}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in flat_axes[image_count:]:
        axis.axis("off")

    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_similarity_heatmap(
    records: list[FaceRecord],
    output_path: str | Path,
) -> None:
    """Calcule et enregistre la heatmap des similarités cosinus galerie×galerie.

    Chaque ligne et colonne représente un visage de l'index. Le chemin de
    l'image et le rang du visage dans cette image apparaissent sur les axes.

    Args:
        records: Visages indexés de la galerie.
        output_path: Fichier image de sortie, généralement
            `similarity_heatmap.png`.

    Raises:
        ValueError: Si les embeddings ne peuvent pas être comparés.
        OSError: Si le fichier de heatmap ne peut pas être écrit.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if records:
        embeddings = np.stack([record.embedding for record in records])
        similarities = cosine_similarity_matrix(embeddings, embeddings)
        labels: list[str] = []
        face_indices: dict[str, int] = {}
        for record in records:
            face_index = face_indices.get(record.image_path, 0)
            face_indices[record.image_path] = face_index + 1
            labels.append(f"{Path(record.image_path).name} #{face_index}")

        figure_width = min(max(8.0, len(records) * 0.35), 24.0)
        figure, axis = plt.subplots(
            figsize=(figure_width, figure_width),
            layout="constrained",
        )
        image = axis.imshow(similarities, cmap="viridis", vmin=-1.0, vmax=1.0)
        axis.set_title("Similarité cosinus entre les visages de la galerie")
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=90, fontsize=7)
        axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
        figure.colorbar(image, ax=axis, label="Similarité cosinus")
    else:
        figure, axis = plt.subplots(figsize=(6, 3))
        axis.text(0.5, 0.5, "Aucun visage indexé", ha="center", va="center")
        axis.set_title("Similarité cosinus galerie × galerie")
        axis.axis("off")

    try:
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def make_contact_sheets(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    columns: int = 5,
) -> list[Path]:
    """Crée une planche de contact par dossier contenant des crops exportés.

    Args:
        report_path: Rapport `report.json` produit par l'organiseur.
        output_dir: Dossier où enregistrer les planches PNG.
        columns: Nombre de crops par rangée.

    Returns:
        Chemins des planches générées, dans l'ordre des dossiers du rapport.

    Raises:
        ValueError: Si le rapport, ses noms ou le nombre de colonnes sont invalides.
        OSError: Si un crop ne peut pas être lu ou une planche enregistrée.
    """
    if columns < 1:
        raise ValueError("columns doit être un entier positif.")

    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("folders"), dict):
        raise TypeError("Le rapport doit contenir un dictionnaire 'folders'.")

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    for folder_name, folder_data in report["folders"].items():
        if (
            not isinstance(folder_name, str)
            or Path(folder_name).name != folder_name
            or not isinstance(folder_data, dict)
            or not isinstance(folder_data.get("crops"), list)
        ):
            raise ValueError("Le rapport contient un dossier ou une liste de crops invalide.")
        crop_entries = folder_data["crops"]
        if not crop_entries:
            continue

        crops: list[tuple[NDArray[np.uint8], str]] = []
        for entry in crop_entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("filename"), str)
                or Path(entry["filename"]).name != entry["filename"]
                or not isinstance(entry.get("source_image"), str)
            ):
                raise ValueError("Le rapport contient une entrée de crop invalide.")
            image = read_image(report_file.parent / folder_name / entry["filename"])
            title = Path(entry["source_image"]).name
            face_index = entry.get("face_index")
            if isinstance(face_index, int):
                title = f"{title} #{face_index}"
            crops.append((image, title))

        rows = math.ceil(len(crops) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(3 * columns, 3.2 * rows),
            squeeze=False,
        )
        for axis, (crop, title) in zip(axes.ravel(), crops, strict=False):
            axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axis.set_title(title, fontsize=8)
            axis.axis("off")
        for axis in axes.ravel()[len(crops) :]:
            axis.axis("off")

        figure.tight_layout()
        output_path = destination_dir / f"{folder_name}.png"
        try:
            figure.savefig(output_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(figure)
        generated_paths.append(output_path)

    return generated_paths
