"""Tests synthétiques de l'expansion de requête et du clustering d'identité."""

from dataclasses import replace

import numpy as np
import pytest

from src.config import AppConfig, load_config
from src.identity import cluster_gallery, expand_query, resolve_identity
from src.types import FaceRecord, MatchResult


def make_record(image_path: str, embedding: list[float]) -> FaceRecord:
    """Construit un visage synthétique pour les tests."""
    return FaceRecord(
        image_path=image_path,
        bbox=(0, 0, 10, 10),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=1.0,
        aligned=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def matching_config(
    *,
    threshold: float = 0.4,
    use_query_expansion: bool = False,
    expansion_top_k: int = 3,
    use_gallery_clustering: bool = False,
    cluster_distance_threshold: float = 0.6,
) -> AppConfig:
    """Retourne la configuration projet avec des options de matching contrôlées."""
    config = load_config()
    return replace(
        config,
        matching=replace(
            config.matching,
            threshold=threshold,
            use_query_expansion=use_query_expansion,
            expansion_top_k=expansion_top_k,
            use_gallery_clustering=use_gallery_clustering,
            cluster_distance_threshold=cluster_distance_threshold,
        ),
    )


def test_expand_query_averages_query_and_top_matched_face_embeddings() -> None:
    """L'expansion prend les k premiers visages, puis renormalise leur moyenne."""
    records = [
        make_record("gallery/a.jpg", [0.0, 1.0]),
        make_record("gallery/b.jpg", [1.0, 0.0]),
        make_record("gallery/c.jpg", [-1.0, 0.0]),
    ]
    top_matches = [
        MatchResult("gallery/b.jpg", 0.9, 0),
        MatchResult("gallery/a.jpg", 0.8, 0),
        MatchResult("gallery/c.jpg", 0.7, 0),
    ]

    expanded = expand_query(np.asarray([1.0, 0.0]), top_matches, records, k=2)

    np.testing.assert_allclose(expanded, [2.0 / np.sqrt(5), 1.0 / np.sqrt(5)])
    assert expanded.dtype == np.float32
    assert np.linalg.norm(expanded) == pytest.approx(1.0)


def test_expand_query_without_matches_returns_normalized_query() -> None:
    """Sans candidats, l'expansion retourne la requête normalisée."""
    expanded = expand_query(np.asarray([3.0, 4.0]), [], [], k=3)

    np.testing.assert_allclose(expanded, [0.6, 0.8])


def test_expand_query_rejects_missing_face_reference() -> None:
    """Un MatchResult incohérent avec l'index déclenche une erreur claire."""
    with pytest.raises(ValueError, match="Impossible de retrouver"):
        expand_query(
            np.asarray([1.0, 0.0]),
            [MatchResult("missing.jpg", 0.9, 0)],
            [],
            k=1,
        )


def test_cluster_gallery_groups_similar_faces_and_keeps_distinct_faces_separate() -> None:
    """Le clustering cosinus regroupe les vecteurs proches et sépare les autres."""
    records = [
        make_record("gallery/a.jpg", [1.0, 0.0]),
        make_record("gallery/b.jpg", [0.98, 0.2]),
        make_record("gallery/c.jpg", [0.0, 1.0]),
    ]

    labels = cluster_gallery(records, distance_threshold=0.1)

    assert labels.shape == (3,)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_cluster_gallery_handles_empty_and_single_face_galleries() -> None:
    """Les tailles 0 et 1 ne requièrent pas d'appel sklearn."""
    assert cluster_gallery([], 0.6).shape == (0,)
    assert cluster_gallery([make_record("one.jpg", [1.0, 0.0])], 0.6).tolist() == [0]


def test_resolve_identity_expansion_is_independent_and_adds_recalled_match() -> None:
    """L'expansion ajoute une image retrouvée, sans dépendre du clustering."""
    records = [
        make_record("gallery/seed.jpg", [0.8, 0.6]),
        make_record("gallery/recovered.jpg", [0.6, 0.8]),
    ]
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    base_results = resolve_identity(
        query,
        records,
        matching_config(threshold=0.65),
    )
    expanded_results = resolve_identity(
        query,
        records,
        matching_config(threshold=0.65, use_query_expansion=True),
    )

    assert [result.image_path for result in base_results] == ["gallery/seed.jpg"]
    assert {result.image_path for result in expanded_results} == {
        "gallery/seed.jpg",
        "gallery/recovered.jpg",
    }


def test_resolve_identity_clustering_is_independent_and_preserves_face_index() -> None:
    """Le cluster ajoute les images liées à la requête même sous le seuil direct."""
    near_embedding = [0.85, float(np.sqrt(1.0 - 0.85**2))]
    records = [
        make_record("gallery/group.jpg", [0.0, 1.0]),
        make_record("gallery/group.jpg", near_embedding),
        make_record("gallery/other.jpg", [0.0, -1.0]),
    ]
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    config = matching_config(
        threshold=0.9,
        use_query_expansion=False,
        use_gallery_clustering=True,
        cluster_distance_threshold=0.2,
    )

    results = resolve_identity(query, records, config)

    assert results == [
        MatchResult(
            "gallery/group.jpg",
            pytest.approx(0.85),
            1,
        )
    ]
    assert (
        resolve_identity(
            query,
            records,
            replace(config, matching=replace(config.matching, use_gallery_clustering=False)),
        )
        == []
    )


def test_resolve_identity_returns_threshold_matches_when_strategies_disabled() -> None:
    """Avec les options désactivées, resolve_identity délègue au matching initial."""
    records = [
        make_record("gallery/match.jpg", [0.8, 0.6]),
        make_record("gallery/no_match.jpg", [0.0, 1.0]),
    ]

    results = resolve_identity(np.asarray([1.0, 0.0]), records, matching_config(threshold=0.7))

    assert [result.image_path for result in results] == ["gallery/match.jpg"]


def test_resolve_identity_empty_gallery_returns_no_matches() -> None:
    """Une galerie vide donne une liste vide même si les options sont actives."""
    config = matching_config(use_query_expansion=True, use_gallery_clustering=True)

    assert resolve_identity(np.asarray([1.0, 0.0]), [], config) == []
