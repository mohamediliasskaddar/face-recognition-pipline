"""Structures de données partagées par le pipeline."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FaceRecord:
    """Visage détecté, aligné et encodé dans une image de la galerie."""

    image_path: str
    bbox: tuple[int, int, int, int]
    landmarks: NDArray[np.float32]
    det_score: float
    aligned: NDArray[np.uint8]
    embedding: NDArray[np.float32]


@dataclass(frozen=True)
class MatchResult:
    """Meilleure correspondance trouvée dans une image de la galerie."""

    image_path: str
    best_score: float
    matched_face_index: int
