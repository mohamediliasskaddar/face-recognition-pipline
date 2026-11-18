"""Clustering des embeddings de visages et rattachement des visages douteux."""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from src.config import AppConfig
from src.quality import is_reliable
from src.types import FaceRecord


def _normalize_embeddings(embeddings: NDArray[np.generic]) -> NDArray[np.float32]:
    """Valide et normalise une matrice d'embeddings ligne par ligne."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Les embeddings doivent être une matrice non vide en dimensions.")
    if not np.isfinite(vectors).all():
        raise ValueError("Les embeddings doivent contenir uniquement des valeurs finies.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("Les embeddings de norme nulle ne peuvent pas être normalisés.")
    if vectors.shape[0] == 0:
        return np.empty(vectors.shape, dtype=np.float32)
    return np.asarray(vectors / norms[:, np.newaxis], dtype=np.float32)


def _validated_labels(
    labels: NDArray[np.generic] | Sequence[int],
    expected_size: int,
) -> NDArray[np.int32]:
    """Valide un label entier par embedding."""
    label_array = np.asarray(labels)
    if label_array.shape != (expected_size,) or label_array.dtype.kind not in {"i", "u"}:
        raise ValueError("labels doit contenir un entier par embedding.")
    if label_array.dtype.kind == "u" and np.any(label_array > np.iinfo(np.int32).max):
        raise ValueError("Les labels dépassent la plage des entiers 32 bits.")
    converted = np.asarray(label_array, dtype=np.int32)
    if np.any(converted < -1):
        raise ValueError("Le label -1 est la seule valeur négative autorisée.")
    return converted


def _validate_cosine_threshold(threshold: float, name: str) -> None:
    """Valide un seuil exprimé en similarité cosinus."""
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError(f"{name} doit être fini et compris entre -1 et 1.")


def cluster_faces(
    embeddings: NDArray[np.generic],
    distance_threshold: float,
) -> NDArray[np.int32]:
    """Regroupe les embeddings fiables avec AgglomerativeClustering cosinus.

    Args:
        embeddings: Matrice `(N, D)` des embeddings fiables.
        distance_threshold: Distance cosinus de coupure, comprise entre 0 et 2.

    Returns:
        Un label entier par visage. Un tableau vide est retourné pour une entrée
        vide ; les labels sont zéro-based.

    Raises:
        ValueError: Si les embeddings ou le seuil sont invalides.
    """
    if not math.isfinite(distance_threshold) or not 0.0 <= distance_threshold <= 2.0:
        raise ValueError("distance_threshold doit être fini et compris entre 0 et 2.")

    normalized = _normalize_embeddings(embeddings)
    face_count = normalized.shape[0]
    if face_count == 0:
        return np.empty(0, dtype=np.int32)
    if face_count == 1:
        return np.zeros(1, dtype=np.int32)

    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(normalized)
    return np.asarray(labels, dtype=np.int32)


def compute_centroids(
    embeddings: NDArray[np.generic],
    labels: NDArray[np.generic] | Sequence[int],
) -> dict[int, NDArray[np.float32]]:
    """Calcule les centroïdes L2 normalisés des clusters, en ignorant `-1`.

    Args:
        embeddings: Matrice `(N, D)` des embeddings.
        labels: Label associé à chaque embedding ; `-1` désigne un visage inconnu.

    Returns:
        Dictionnaire trié par label associant chaque cluster à son centroïde.

    Raises:
        ValueError: Si les embeddings ou les labels sont invalides.
    """
    normalized = _normalize_embeddings(embeddings)
    label_array = _validated_labels(labels, normalized.shape[0])
    centroids: dict[int, NDArray[np.float32]] = {}
    for label in np.unique(label_array):
        if label == -1:
            continue
        centroid = normalized[label_array == label].mean(axis=0)
        norm = float(np.linalg.norm(centroid.astype(np.float64)))
        if norm == 0.0:
            raise ValueError(f"Le centroïde du cluster {label} a une norme nulle.")
        centroids[int(label)] = np.asarray(centroid / norm, dtype=np.float32)
    return centroids


def attach_low_quality(
    embeddings_low: NDArray[np.generic],
    centroids: Mapping[int, NDArray[np.generic]],
    threshold: float,
) -> NDArray[np.int32]:
    """Rattache chaque visage douteux au centroïde cosinus le plus proche.

    Les visages sous le seuil, ou tous les visages lorsque la liste des
    centroïdes est vide, reçoivent le label `-1`.
    """
    _validate_cosine_threshold(threshold, "threshold")
    normalized = _normalize_embeddings(embeddings_low)
    if normalized.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    if not centroids:
        return np.full(normalized.shape[0], -1, dtype=np.int32)

    labels = sorted(centroids)
    if any(label < 0 for label in labels):
        raise ValueError("Les centroïdes ne peuvent pas utiliser le label réservé -1.")
    centroid_matrix = _normalize_embeddings(np.stack([centroids[label] for label in labels]))
    if centroid_matrix.shape[1] != normalized.shape[1]:
        raise ValueError("Les dimensions des embeddings et des centroïdes doivent correspondre.")

    similarities = normalized @ centroid_matrix.T
    best_indices = np.argmax(similarities, axis=1)
    best_scores = similarities[np.arange(len(normalized)), best_indices]
    attached = np.full(normalized.shape[0], -1, dtype=np.int32)
    accepted = best_scores >= threshold
    attached[accepted] = np.asarray(labels, dtype=np.int32)[best_indices[accepted]]
    return attached


def merge_close_clusters(
    centroids: Mapping[int, NDArray[np.generic]],
    labels: NDArray[np.generic] | Sequence[int],
    merge_threshold: float,
) -> tuple[dict[int, NDArray[np.float32]], NDArray[np.int32]]:
    """Fusionne transitivement les clusters dont les centroïdes sont assez proches.

    Les nouveaux labels sont zéro-based et ordonnés selon le plus petit label
    d'origine. Le centroïde fusionné est la moyenne des centroïdes unitaires,
    pondérée par le nombre de visages de chaque cluster. Les labels `-1` restent
    inconnus.
    """
    _validate_cosine_threshold(merge_threshold, "merge_threshold")
    label_array = _validated_labels(labels, len(labels))
    cluster_labels = sorted(int(label) for label in np.unique(label_array) if label >= 0)
    if not cluster_labels:
        if centroids:
            raise ValueError("Aucun label de cluster ne correspond aux centroïdes fournis.")
        return {}, label_array.copy()
    if set(cluster_labels) != set(centroids):
        raise ValueError(
            "Les labels présents et les centroïdes doivent désigner les mêmes clusters."
        )

    centroid_matrix = _normalize_embeddings(
        np.stack([centroids[label] for label in cluster_labels])
    )
    parent = {label: label for label in cluster_labels}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for first_index, first_label in enumerate(cluster_labels):
        for second_index in range(first_index + 1, len(cluster_labels)):
            if float(np.dot(centroid_matrix[first_index], centroid_matrix[second_index])) >= (
                merge_threshold
            ):
                union(first_label, cluster_labels[second_index])

    groups: dict[int, list[int]] = {}
    for label in cluster_labels:
        groups.setdefault(find(label), []).append(label)

    remapped_labels = np.full(label_array.shape, -1, dtype=np.int32)
    merged_centroids: dict[int, NDArray[np.float32]] = {}
    for new_label, group in enumerate(sorted(groups.values(), key=min)):
        weights = np.asarray(
            [np.count_nonzero(label_array == label) for label in group],
            dtype=np.float64,
        )
        group_indices = [cluster_labels.index(label) for label in group]
        weighted_centroid = np.average(
            centroid_matrix[group_indices],
            axis=0,
            weights=weights,
        )
        norm = float(np.linalg.norm(weighted_centroid.astype(np.float64)))
        if norm == 0.0:
            raise ValueError("La fusion produit un centroïde de norme nulle.")
        merged_centroids[new_label] = np.asarray(weighted_centroid / norm, dtype=np.float32)
        for old_label in group:
            remapped_labels[label_array == old_label] = new_label

    return merged_centroids, remapped_labels


def cluster_face_records(
    records: Sequence[FaceRecord],
    cfg: AppConfig,
    *,
    distance_threshold: float | None = None,
) -> NDArray[np.int32]:
    """Clusterise les visages fiables puis rattache les visages douteux.

    Les labels de sortie suivent l'ordre de `records`; tout visage douteux sans
    centroïde assez proche reçoit le label `-1`.
    """
    if not records:
        return np.empty(0, dtype=np.int32)

    reliable_mask = np.asarray([is_reliable(record, cfg) for record in records], dtype=np.bool_)
    reliable_indices = np.flatnonzero(reliable_mask)
    low_quality_indices = np.flatnonzero(~reliable_mask)
    final_labels = np.full(len(records), -1, dtype=np.int32)
    if reliable_indices.size == 0:
        return final_labels

    reliable_embeddings = np.stack([records[index].embedding for index in reliable_indices])
    cluster_labels = cluster_faces(
        reliable_embeddings,
        cfg.clustering.distance_threshold if distance_threshold is None else distance_threshold,
    )
    centroids = compute_centroids(reliable_embeddings, cluster_labels)
    centroids, cluster_labels = merge_close_clusters(
        centroids,
        cluster_labels,
        cfg.clustering.merge_threshold,
    )
    final_labels[reliable_indices] = cluster_labels

    if low_quality_indices.size:
        low_embeddings = np.stack([records[index].embedding for index in low_quality_indices])
        final_labels[low_quality_indices] = attach_low_quality(
            low_embeddings,
            centroids,
            cfg.clustering.attach_threshold,
        )
    return final_labels
