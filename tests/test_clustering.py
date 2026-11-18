"""Tests synthétiques du clustering des embeddings faciaux."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.clustering import (
    attach_low_quality,
    cluster_faces,
    compute_centroids,
    merge_close_clusters,
)


def test_cluster_faces_finds_three_groups_without_known_cluster_count() -> None:
    """Agglomerative cosinus retrouve trois groupes synthétiques séparés."""
    rng = np.random.default_rng(42)
    centers = np.eye(3, 512, dtype=np.float32)
    embeddings = np.vstack(
        [center + rng.normal(0.0, 0.01, size=(8, 512)).astype(np.float32) for center in centers]
    )

    labels = cluster_faces(embeddings, distance_threshold=0.25)

    assert len(np.unique(labels)) == 3
    for group_index in range(3):
        assert len(np.unique(labels[group_index * 8 : (group_index + 1) * 8])) == 1


def test_cluster_faces_handles_empty_and_single_face_inputs() -> None:
    """Une entrée vide reste vide et un visage isolé forme son propre cluster."""
    assert cluster_faces(np.empty((0, 4), dtype=np.float32), 0.6).size == 0
    np.testing.assert_array_equal(
        cluster_faces(np.asarray([[3.0, 4.0]], dtype=np.float32), 0.6),
        np.asarray([0], dtype=np.int32),
    )


def test_cluster_faces_keeps_or_merges_two_very_close_identities_by_threshold() -> None:
    """Un faible écart peut être séparé ou fusionné selon le seuil choisi."""
    embeddings = np.asarray([[1.0, 0.0], [0.98, 0.2]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    separate_labels = cluster_faces(embeddings, distance_threshold=0.01)
    merged_labels = cluster_faces(embeddings, distance_threshold=0.1)

    assert len(np.unique(separate_labels)) == 2
    assert len(np.unique(merged_labels)) == 1


def test_compute_centroids_normalizes_means_and_ignores_unknown_faces() -> None:
    """Les centroïdes sont normalisés et le label -1 en est exclu."""
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, -1], dtype=np.int32)

    centroids = compute_centroids(embeddings, labels)

    assert set(centroids) == {0, 1}
    assert np.linalg.norm(centroids[0]) == pytest.approx(1.0)
    assert np.linalg.norm(centroids[1]) == pytest.approx(1.0)
    assert centroids[0][0] == pytest.approx(1.0 / np.sqrt(2.0))


def test_attach_low_quality_assigns_closest_centroid_or_unknown() -> None:
    """Les visages douteux sous le seuil restent unknown."""
    centroids = {
        3: np.asarray([1.0, 0.0], dtype=np.float32),
        7: np.asarray([0.0, 1.0], dtype=np.float32),
    }
    low_embeddings = np.asarray(
        [[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]],
        dtype=np.float32,
    )

    labels = attach_low_quality(low_embeddings, centroids, threshold=0.8)

    np.testing.assert_array_equal(labels, np.asarray([3, 7, -1], dtype=np.int32))


def test_attach_low_quality_marks_all_faces_unknown_without_centroids() -> None:
    """Sans centroïdes, chaque visage douteux reçoit le label réservé -1."""
    labels = attach_low_quality(np.ones((2, 3), dtype=np.float32), {}, threshold=0.35)

    np.testing.assert_array_equal(labels, np.asarray([-1, -1], dtype=np.int32))


def test_merge_close_clusters_merges_labels_and_recomputes_centroids() -> None:
    """Les clusters proches sont fusionnés et les inconnus restent inchangés."""
    centroids = {
        2: np.asarray([1.0, 0.0], dtype=np.float32),
        5: np.asarray([0.98, 0.2], dtype=np.float32),
        8: np.asarray([0.0, 1.0], dtype=np.float32),
    }
    labels = np.asarray([2, 2, 5, 8, -1], dtype=np.int32)

    merged_centroids, merged_labels = merge_close_clusters(
        centroids,
        labels,
        merge_threshold=0.9,
    )

    assert set(merged_centroids) == {0, 1}
    np.testing.assert_array_equal(merged_labels, np.asarray([0, 0, 0, 1, -1]))
    assert np.linalg.norm(merged_centroids[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("embeddings", "threshold"),
    [
        (np.asarray([[0.0, 0.0]], dtype=np.float32), 0.6),
        (np.asarray([[np.nan, 1.0]], dtype=np.float32), 0.6),
        (np.asarray([[1.0, 0.0]], dtype=np.float32), 2.1),
    ],
)
def test_cluster_faces_rejects_invalid_embeddings_or_threshold(
    embeddings: NDArray[np.float32],
    threshold: float,
) -> None:
    """Les entrées invalides sont signalées sans produire de labels trompeurs."""
    with pytest.raises(ValueError):
        cluster_faces(embeddings, threshold)
