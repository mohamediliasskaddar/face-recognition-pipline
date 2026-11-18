"""Tests de l'évaluation de la fiabilité des visages."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import load_config
from src.quality import is_reliable, quality_score
from src.types import FaceRecord


def make_face(
    *,
    det_score: float = 0.9,
    bbox: tuple[int, int, int, int] = (0, 0, 80, 80),
    aligned: NDArray[np.uint8] | None = None,
) -> FaceRecord:
    """Construit un visage synthétique dont le crop est net par défaut."""
    if aligned is None:
        checkerboard = (np.indices((112, 112)).sum(axis=0) % 2 * 255).astype(np.uint8)
        aligned = np.repeat(checkerboard[:, :, None], 3, axis=2)
    return FaceRecord(
        image_path="face.jpg",
        bbox=bbox,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=det_score,
        aligned=aligned,
        embedding=np.zeros(512, dtype=np.float32),
    )


def test_quality_score_is_lowest_normalized_quality_measure() -> None:
    """Le score reflète le critère le plus éloigné de son seuil."""
    face = make_face(bbox=(0, 0, 40, 80))

    assert quality_score(face, load_config()) == pytest.approx(0.8)
    assert not is_reliable(face, load_config())


@pytest.mark.parametrize(
    "face",
    [
        make_face(det_score=0.5),
        make_face(bbox=(0, 0, 40, 80)),
        make_face(aligned=np.full((112, 112, 3), 127, dtype=np.uint8)),
    ],
)
def test_is_reliable_rejects_faces_below_any_quality_threshold(face: FaceRecord) -> None:
    """Un seul critère sous son seuil suffit à classer un visage comme douteux."""
    assert not is_reliable(face, load_config())


def test_is_reliable_accepts_a_face_above_all_quality_thresholds() -> None:
    """Un visage assez grand, net et bien détecté est considéré fiable."""
    assert is_reliable(make_face(), load_config())


def test_quality_score_rejects_invalid_bounding_box() -> None:
    """Une boîte dégénérée est signalée plutôt que convertie en score valide."""
    with pytest.raises(ValueError, match="largeur et une hauteur positives"):
        quality_score(make_face(bbox=(0, 0, 0, 80)), load_config())
