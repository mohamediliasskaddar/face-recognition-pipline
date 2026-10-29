"""Tests de comparaison et d'agrégation des correspondances."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.matching import cosine_similarity_matrix, match_images
from src.types import FaceRecord, MatchResult


def make_record(image_path: str, embedding: NDArray[np.float32]) -> FaceRecord:
    """Construit un enregistrement minimal pour les tests de correspondance."""
    return FaceRecord(
        image_path=image_path,
        bbox=(0, 0, 10, 10),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=1.0,
        aligned=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=embedding,
    )


def test_cosine_similarity_matrix_is_vectorized_and_normalizes_inputs() -> None:
    """La matrice compare toutes les paires et supporte des vecteurs non unitaires."""
    queries = np.asarray([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    gallery = np.asarray([[2.0, 0.0], [1.0, 1.0], [-4.0, 0.0]], dtype=np.float32)

    similarities = cosine_similarity_matrix(queries, gallery)

    assert similarities.shape == (2, 3)
    np.testing.assert_allclose(
        similarities,
        [[1.0, 1.0 / np.sqrt(2.0), -1.0], [0.0, 1.0 / np.sqrt(2.0), 0.0]],
        atol=1e-6,
    )


def test_match_images_uses_best_face_and_sorts_images_by_score() -> None:
    """Une image multi-visage est scorée par son meilleur visage."""
    records = [
        make_record("gallery/a.jpg", np.asarray([0.0, 1.0], dtype=np.float32)),
        make_record("gallery/b.jpg", np.asarray([0.8, 0.6], dtype=np.float32)),
        make_record("gallery/a.jpg", np.asarray([1.0, 0.0], dtype=np.float32)),
        make_record("gallery/c.jpg", np.asarray([-1.0, 0.0], dtype=np.float32)),
    ]

    matches = match_images(np.asarray([1.0, 0.0], dtype=np.float32), records, 0.5)

    assert matches == [
        MatchResult("gallery/a.jpg", 1.0, 1),
        MatchResult("gallery/b.jpg", pytest.approx(0.8), 0),
    ]


def test_match_images_reports_index_of_best_face_within_image() -> None:
    """L'index retourné est celui du visage gagnant dans son image."""
    records = [
        make_record("gallery/group.jpg", np.asarray([0.0, 1.0], dtype=np.float32)),
        make_record("gallery/group.jpg", np.asarray([1.0, 0.0], dtype=np.float32)),
    ]

    matches = match_images(np.asarray([1.0, 0.0], dtype=np.float32), records, 0.9)

    assert matches == [MatchResult("gallery/group.jpg", 1.0, 1)]


def test_match_images_includes_scores_equal_to_threshold() -> None:
    """Le seuil est inclusif."""
    records = [make_record("gallery/exact.jpg", np.asarray([0.6, 0.8], dtype=np.float32))]

    matches = match_images(np.asarray([1.0, 0.0], dtype=np.float32), records, 0.6)

    assert len(matches) == 1
    assert matches[0].best_score == pytest.approx(0.6)


def test_match_images_returns_empty_for_empty_gallery() -> None:
    """Une galerie sans visage ne produit aucune correspondance."""
    assert match_images(np.asarray([1.0, 0.0], dtype=np.float32), [], 0.4) == []


@pytest.mark.parametrize(
    ("query", "gallery"),
    [
        (np.asarray([0.0, 0.0]), np.asarray([[1.0, 0.0]])),
        (np.asarray([1.0, 0.0]), np.asarray([[0.0, 0.0]])),
        (np.asarray([1.0, 0.0]), np.asarray([[1.0]])),
        (np.asarray([1.0, np.nan]), np.asarray([[1.0, 0.0]])),
        (np.ones((1, 1, 2)), np.ones((1, 2))),
        (np.ones(2), np.ones((1, 2, 1))),
    ],
)
def test_cosine_similarity_matrix_rejects_invalid_embeddings(
    query: NDArray[np.float64],
    gallery: NDArray[np.float64],
) -> None:
    """Les formes invalides, valeurs non finies et vecteurs nuls sont explicites."""
    with pytest.raises(ValueError):
        cosine_similarity_matrix(query, gallery)
