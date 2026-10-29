"""Évaluation des résultats et calibration du seuil."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_curve

from src.matching import cosine_similarity_matrix, match_images
from src.types import FaceRecord, MatchResult

DEFAULT_THRESHOLDS = np.linspace(0.2, 0.7, 51, dtype=np.float64)


def load_labels(labels_csv: str | Path) -> dict[str, frozenset[str]]:
    """Lit les identités annotées, en acceptant plusieurs IDs séparés par `;`.

    Args:
        labels_csv: Fichier CSV avec les colonnes `filename,person_id`.

    Returns:
        Association entre chaque nom de fichier et son ou ses IDs d'identité.

    Raises:
        ValueError: Si le CSV ne respecte pas le format attendu ou ne contient
            aucune annotation exploitable.
        OSError: Si le fichier ne peut pas être lu.
    """
    labels: dict[str, set[str]] = {}
    with Path(labels_csv).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"filename", "person_id"}.issubset(reader.fieldnames):
            raise ValueError("Le CSV doit contenir les colonnes 'filename' et 'person_id'.")

        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            person_ids = {
                person_id.strip()
                for person_id in (row.get("person_id") or "").split(";")
                if person_id.strip()
            }
            if not filename or not person_ids:
                raise ValueError(
                    f"Ligne {row_number} du CSV : filename et person_id doivent être renseignés."
                )
            if Path(filename).name != filename:
                raise ValueError(
                    f"Ligne {row_number} du CSV : filename doit être un nom de fichier, "
                    "pas un chemin."
                )
            labels.setdefault(filename, set()).update(person_ids)

    if not labels:
        raise ValueError(f"Aucune annotation trouvée dans '{labels_csv}'.")
    return {filename: frozenset(person_ids) for filename, person_ids in labels.items()}


def _metrics(
    results: Sequence[MatchResult],
    labels: dict[str, frozenset[str]],
    query_person_id: str,
    query_image_path: str | Path | None,
) -> dict[str, float | int]:
    """Calcule les métriques d'une identité à partir d'annotations déjà chargées."""
    query_filename = Path(query_image_path).name if query_image_path is not None else None
    expected = {
        filename
        for filename, person_ids in labels.items()
        if query_person_id in person_ids and filename != query_filename
    }
    if not expected:
        raise ValueError(
            f"Aucune image positive pour l'identité '{query_person_id}' "
            "après exclusion de l'image requête."
        )

    predicted = {
        Path(result.image_path).name
        for result in results
        if Path(result.image_path).name != query_filename
    }
    true_positives = len(predicted & expected)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate(
    results: Sequence[MatchResult],
    labels_csv: str | Path,
    query_person_id: str,
    query_image_path: str | Path | None = None,
) -> dict[str, float | int]:
    """Calcule précision, rappel et F1 pour une identité requête.

    L'image requête est exclue des prédictions et de la vérité terrain si son
    chemin est fourni, afin qu'elle ne compte pas comme un résultat trivial.

    Args:
        results: Résultats de `match_images` ou `resolve_identity`.
        labels_csv: CSV des identités de référence.
        query_person_id: Identité recherchée.
        query_image_path: Image galerie utilisée comme requête, le cas échéant.

    Returns:
        Métriques et nombres de vrais/faux positifs et négatifs.

    Raises:
        ValueError: Si l'identité est inconnue ou n'a aucune image positive
            après exclusion de la requête.
    """
    if not query_person_id.strip():
        raise ValueError("query_person_id ne peut pas être vide.")
    labels = load_labels(labels_csv)
    if not any(query_person_id in person_ids for person_ids in labels.values()):
        raise ValueError(f"L'identité '{query_person_id}' est absente du CSV.")
    return _metrics(results, labels, query_person_id, query_image_path)


def _query_samples(
    records: Sequence[FaceRecord],
    labels: dict[str, frozenset[str]],
) -> list[tuple[str, str, NDArray[np.float32]]]:
    """Choisit déterministement une image requête indexée par identité évaluable."""
    records_by_filename: dict[str, list[FaceRecord]] = {}
    for record in records:
        records_by_filename.setdefault(Path(record.image_path).name, []).append(record)

    identities = sorted({person_id for ids in labels.values() for person_id in ids})
    samples: list[tuple[str, str, NDArray[np.float32]]] = []
    for person_id in identities:
        available_images = sorted(
            filename
            for filename, person_ids in labels.items()
            if person_id in person_ids and filename in records_by_filename
        )
        if len(available_images) < 2:
            continue
        query_filename = available_images[0]
        samples.append(
            (
                person_id,
                records_by_filename[query_filename][0].image_path,
                records_by_filename[query_filename][0].embedding,
            )
        )
    if not samples:
        raise ValueError(
            "Aucune identité n'a au moins deux images annotées avec un visage indexé; "
            "il faut une image requête et au moins une image positive distincte."
        )
    return samples


def sweep_threshold(
    records: Sequence[FaceRecord],
    labels_csv: str | Path,
    thresholds: Sequence[float] | NDArray[np.floating] | None = None,
) -> list[dict[str, float | int]]:
    """Évalue une plage de seuils et moyenne les métriques par identité.

    Pour chaque identité ayant au moins deux images indexées, la première image
    par ordre alphabétique sert de requête; l'image requête n'est pas comptée
    dans les prédictions ni dans la vérité terrain.

    Args:
        records: Visages indexés de la galerie.
        labels_csv: CSV des identités de référence.
        thresholds: Seuils cosinus à tester; par défaut, 51 valeurs de 0.20 à 0.70.

    Returns:
        Une ligne par seuil avec les métriques macro et le nombre d'identités.

    Raises:
        ValueError: Si les seuils sont invalides ou si aucune identité n'est évaluable.
    """
    threshold_values = np.asarray(
        DEFAULT_THRESHOLDS if thresholds is None else thresholds,
        dtype=np.float64,
    )
    if (
        threshold_values.ndim != 1
        or threshold_values.size == 0
        or not np.isfinite(threshold_values).all()
        or np.any((threshold_values < -1.0) | (threshold_values > 1.0))
    ):
        raise ValueError("Les seuils doivent être une séquence non vide comprise entre -1 et 1.")

    labels = load_labels(labels_csv)
    samples = _query_samples(records, labels)
    curve: list[dict[str, float | int]] = []
    gallery_records = list(records)
    for threshold in threshold_values:
        per_identity = [
            _metrics(
                match_images(embedding, gallery_records, float(threshold)),
                labels,
                person_id,
                query_path,
            )
            for person_id, query_path, embedding in samples
        ]
        curve.append(
            {
                "threshold": float(threshold),
                "precision": float(np.mean([metric["precision"] for metric in per_identity])),
                "recall": float(np.mean([metric["recall"] for metric in per_identity])),
                "f1": float(np.mean([metric["f1"] for metric in per_identity])),
                "n_queries": len(per_identity),
            }
        )
    return curve


def roc_curve_pairs(
    records: Sequence[FaceRecord],
    labels_csv: str | Path,
) -> dict[str, NDArray[np.float64] | float | int]:
    """Calcule ROC et EER sur les paires de visages d'images distinctes.

    Une paire est positive lorsque les deux images partagent au moins un ID
    annoté. Les paires de visages appartenant à la même image sont ignorées.

    Args:
        records: Visages indexés de la galerie.
        labels_csv: CSV des identités de référence.

    Returns:
        Taux de faux positifs, taux de vrais positifs, seuils ROC, EER,
        seuil EER et nombres de paires positives/négatives.

    Raises:
        ValueError: Si les annotations ou les enregistrements ne fournissent pas
            au moins une paire positive et une paire négative.
    """
    labels = load_labels(labels_csv)
    labelled_records = [record for record in records if Path(record.image_path).name in labels]
    if len(labelled_records) < 2:
        raise ValueError("La ROC exige au moins deux visages annotés dans des images distinctes.")
    embeddings = np.stack([record.embedding for record in labelled_records])
    similarities = cosine_similarity_matrix(embeddings, embeddings)
    pair_scores: list[float] = []
    pair_labels: list[bool] = []
    for first_index, first in enumerate(labelled_records):
        first_filename = Path(first.image_path).name
        for second_index in range(first_index + 1, len(labelled_records)):
            second = labelled_records[second_index]
            second_filename = Path(second.image_path).name
            if first_filename == second_filename:
                continue
            pair_scores.append(float(similarities[first_index, second_index]))
            pair_labels.append(bool(labels[first_filename] & labels[second_filename]))

    targets = np.asarray(pair_labels, dtype=np.bool_)
    scores = np.asarray(pair_scores, dtype=np.float64)
    positive_count = int(np.count_nonzero(targets))
    negative_count = int(targets.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "La ROC exige au moins une paire positive et une paire négative "
            "entre des images distinctes."
        )

    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        targets,
        scores,
        drop_intermediate=False,
    )
    difference = true_positive_rate - (1.0 - false_positive_rate)
    crossing_indices = np.flatnonzero(difference >= 0.0)
    if crossing_indices.size == 0:
        eer_index = int(np.argmin(np.abs(difference)))
        eer = float((false_positive_rate[eer_index] + 1.0 - true_positive_rate[eer_index]) / 2.0)
        eer_threshold = float(roc_thresholds[eer_index])
    else:
        upper_index = int(crossing_indices[0])
        if upper_index == 0:
            eer = float((false_positive_rate[0] + 1.0 - true_positive_rate[0]) / 2.0)
            eer_threshold = float(roc_thresholds[0])
        else:
            lower_index = upper_index - 1
            lower_difference = difference[lower_index]
            upper_difference = difference[upper_index]
            fraction = -lower_difference / (upper_difference - lower_difference)
            eer = float(
                false_positive_rate[lower_index]
                + fraction * (false_positive_rate[upper_index] - false_positive_rate[lower_index])
            )
            eer_threshold = (
                float(
                    roc_thresholds[lower_index]
                    + fraction * (roc_thresholds[upper_index] - roc_thresholds[lower_index])
                )
                if np.isfinite(roc_thresholds[lower_index])
                else float(roc_thresholds[upper_index])
            )

    return {
        "fpr": false_positive_rate.astype(np.float64),
        "tpr": true_positive_rate.astype(np.float64),
        "thresholds": roc_thresholds.astype(np.float64),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "n_positive_pairs": positive_count,
        "n_negative_pairs": negative_count,
    }
