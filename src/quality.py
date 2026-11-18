"""Évaluation de la fiabilité des visages détectés."""

import cv2
import numpy as np

from src.config import AppConfig
from src.types import FaceRecord


def quality_score(face: FaceRecord, cfg: AppConfig) -> float:
    """Retourne le plus faible des rapports entre mesures et seuils de qualité.

    Le score vaut au moins `1.0` si la confiance, la plus petite dimension de
    la boîte et la netteté atteignent toutes leurs seuils configurés.

    Args:
        face: Enregistrement du visage, incluant le crop aligné BGR.
        cfg: Configuration contenant les seuils de qualité.

    Returns:
        Score normalisé par rapport aux seuils, inférieur à `1.0` si au moins
        un critère n'atteint pas son seuil.

    Raises:
        ValueError: Si les données du visage sont invalides.
    """
    bbox = np.asarray(face.bbox, dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError("La boîte du visage doit contenir quatre coordonnées finies.")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        raise ValueError("La boîte du visage doit avoir une largeur et une hauteur positives.")

    if not np.isfinite(face.det_score) or not 0.0 <= face.det_score <= 1.0:
        raise ValueError("Le score du détecteur doit être fini et compris entre 0 et 1.")

    aligned_face = face.aligned
    if (
        not isinstance(aligned_face, np.ndarray)
        or aligned_face.dtype != np.uint8
        or aligned_face.ndim != 3
        or aligned_face.shape[2] != 3
        or min(aligned_face.shape[:2]) < 3
    ):
        raise ValueError("Le crop aligné doit être une image BGR uint8 valide.")

    gray_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    return min(
        face.det_score / cfg.quality.min_det_score,
        min(width, height) / cfg.quality.min_face_size,
        sharpness / cfg.quality.min_sharpness,
    )


def is_reliable(face: FaceRecord, cfg: AppConfig) -> bool:
    """Indique si les trois critères configurés sont atteints."""
    return quality_score(face, cfg) >= 1.0
