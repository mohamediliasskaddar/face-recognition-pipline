"""Tests de métriques, balayage des seuils et courbe ROC."""

import csv
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import evaluate, load_labels, roc_curve_pairs, sweep_threshold
from src.types import FaceRecord, MatchResult


def make_record(image_path: str, embedding: list[float]) -> FaceRecord:
    """Construit un visage synthétique pour les tests d'évaluation."""
    return FaceRecord(
        image_path=image_path,
        bbox=(0, 0, 10, 10),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=1.0,
        aligned=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def write_labels(path: Path, rows: list[tuple[str, str]]) -> Path:
    """Écrit un petit fichier de vérité terrain."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "person_id"])
        writer.writerows(rows)
    return path


def test_load_labels_merges_repeated_rows_and_semicolon_ids(tmp_path: Path) -> None:
    """Les lignes répétées sont fusionnées et un fichier accepte plusieurs IDs."""
    labels_path = write_labels(
        tmp_path / "labels.csv",
        [("group.jpg", "P01; P02"), ("group.jpg", "P03")],
    )

    labels = load_labels(labels_path)

    assert labels == {"group.jpg": frozenset({"P01", "P02", "P03"})}


def test_evaluate_excludes_query_and_computes_precision_recall_f1(
    tmp_path: Path,
) -> None:
    """L'image requête ne compte pas comme un vrai positif trivial."""
    labels_path = write_labels(
        tmp_path / "labels.csv",
        [("query.jpg", "P01"), ("positive.jpg", "P01"), ("negative.jpg", "P02")],
    )
    results = [
        MatchResult("gallery/query.jpg", 1.0, 0),
        MatchResult("gallery/positive.jpg", 0.8, 0),
        MatchResult("gallery/negative.jpg", 0.7, 0),
    ]

    metrics = evaluate(results, labels_path, "P01", "gallery/query.jpg")

    assert metrics == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": pytest.approx(2.0 / 3.0),
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 0,
    }


def test_evaluate_rejects_unknown_identity_and_query_without_positive_target(
    tmp_path: Path,
) -> None:
    """Les métriques n'acceptent pas une vérité terrain vide ou une ID inconnue."""
    labels_path = write_labels(
        tmp_path / "labels.csv",
        [("query.jpg", "P01"), ("positive.jpg", "P01")],
    )

    with pytest.raises(ValueError, match="absente du CSV"):
        evaluate([], labels_path, "P02")
    query_only_labels = write_labels(
        tmp_path / "query_only.csv",
        [("query.jpg", "P01")],
    )
    with pytest.raises(ValueError, match="Aucune image positive"):
        evaluate([], query_only_labels, "P01", "query.jpg")


def test_sweep_threshold_averages_over_query_identities_and_excludes_query(
    tmp_path: Path,
) -> None:
    """Le balayage compare plusieurs seuils pour chaque identité évaluable."""
    labels_path = write_labels(
        tmp_path / "labels.csv",
        [
            ("a-query.jpg", "P01"),
            ("b-positive.jpg", "P01"),
            ("negative.jpg", "P02"),
        ],
    )
    records = [
        make_record("gallery/a-query.jpg", [1.0, 0.0]),
        make_record("gallery/b-positive.jpg", [0.8, 0.6]),
        make_record("gallery/negative.jpg", [0.0, 1.0]),
    ]

    curve = sweep_threshold(records, labels_path, thresholds=[0.5, 0.9])

    assert curve == [
        {
            "threshold": 0.5,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "n_queries": 1,
        },
        {
            "threshold": 0.9,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_queries": 1,
        },
    ]


def test_sweep_threshold_rejects_invalid_thresholds(tmp_path: Path) -> None:
    """Le balayage refuse une plage vide ou hors de l'intervalle cosinus."""
    labels_path = write_labels(
        tmp_path / "labels.csv",
        [("query.jpg", "P01"), ("positive.jpg", "P01")],
    )
    records = [
        make_record("query.jpg", [1.0, 0.0]),
        make_record("positive.jpg", [0.8, 0.6]),
    ]

    with pytest.raises(ValueError, match="seuils"):
        sweep_threshold(records, labels_path, thresholds=[])
    with pytest.raises(ValueError, match="seuils"):
        sweep_threshold(records, labels_path, thresholds=[1.1])


def test_roc_curve_pairs_computes_positive_negative_pairs_and_eer(
    tmp_path: Path,
) -> None:
    """La ROC compare uniquement des images distinctes avec deux classes."""
    labels_path = write_labels(
        tmp_path / "labels.csv",
        [("query.jpg", "P01"), ("positive.jpg", "P01"), ("negative.jpg", "P02")],
    )
    records = [
        make_record("gallery/query.jpg", [1.0, 0.0]),
        make_record("gallery/positive.jpg", [0.8, 0.6]),
        make_record("gallery/negative.jpg", [0.0, 1.0]),
    ]

    roc = roc_curve_pairs(records, labels_path)

    assert roc["n_positive_pairs"] == 1
    assert roc["n_negative_pairs"] == 2
    assert 0.0 <= roc["eer"] <= 1.0
    assert np.asarray(roc["fpr"]).shape == np.asarray(roc["tpr"]).shape


def test_roc_curve_pairs_requires_both_pair_classes(tmp_path: Path) -> None:
    """La ROC échoue explicitement quand aucune paire positive n'est annotée."""
    labels_path = write_labels(
        tmp_path / "labels.csv",
        [("first.jpg", "P01"), ("second.jpg", "P02")],
    )
    records = [
        make_record("first.jpg", [1.0, 0.0]),
        make_record("second.jpg", [0.0, 1.0]),
    ]

    with pytest.raises(ValueError, match="paire positive"):
        roc_curve_pairs(records, labels_path)
