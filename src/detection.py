"""Détection des visages avec InsightFace."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from src.config import AppConfig, ModelConfig, load_config


class RecognitionModel(Protocol):
    """Interface du modèle ArcFace chargé par FaceAnalysis."""

    def get_feat(self, imgs: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Calcule l'embedding brut d'un visage aligné."""


class DetectedFace(Protocol):
    """Attributs InsightFace utilisés par le pipeline après une détection."""

    bbox: NDArray[np.float32]
    kps: NDArray[np.float32]
    det_score: float


class FaceAnalysisBackend(Protocol):
    """Interface minimale du moteur FaceAnalysis utilisé par le détecteur."""

    models: dict[str, RecognitionModel]

    def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
        """Prépare les modèles InsightFace pour l'inférence."""

    def get(self, img_bgr: NDArray[np.uint8]) -> list[DetectedFace]:
        """Détecte les visages présents dans une image BGR."""


def _create_analysis(model_config: ModelConfig) -> FaceAnalysisBackend:
    """Construit et prépare une instance InsightFace."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError as error:
        raise ImportError(
            "InsightFace est requis pour la détection. Installez les dépendances "
            "du projet depuis requirements.txt."
        ) from error

    analysis = FaceAnalysis(
        name=model_config.name,
        providers=list(model_config.providers),
    )
    analysis.prepare(ctx_id=-1, det_size=model_config.det_size)
    return analysis


class FaceDetector:
    """Détecte les visages et filtre les résultats selon la configuration."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        analysis: FaceAnalysisBackend | None = None,
    ) -> None:
        """Crée et prépare une seule fois le moteur InsightFace.

        Args:
            config: Configuration du pipeline. La configuration du projet est
                chargée si elle n'est pas fournie.
            analysis: Moteur déjà chargé à partager avec d'autres composants.
        """
        self._config = config if config is not None else load_config()
        self._analysis = (
            analysis if analysis is not None else _create_analysis(self._config.model)
        )

    @property
    def analysis(self) -> FaceAnalysisBackend:
        """Retourne le moteur pour le partager avec l'embedder."""
        return self._analysis

    def detect(self, img_bgr: NDArray[np.uint8]) -> list[DetectedFace]:
        """Détecte les visages et retourne les détections valides par aire décroissante.

        Un visage est conservé si son score est au moins `min_det_score` et si
        sa plus petite dimension est au moins `min_face_size`.

        Args:
            img_bgr: Image couleur BGR de type uint8, telle que retournée par
                `read_image`.

        Returns:
            Visages détectés, filtrés et triés de la plus grande aire à la plus petite.

        Raises:
            TypeError: Si l'entrée n'est pas un tableau numpy uint8.
            ValueError: Si l'image est vide ou ne contient pas trois canaux couleur,
                ou si InsightFace retourne une boîte invalide.
        """
        if not isinstance(img_bgr, np.ndarray):
            raise TypeError("L'image doit être un tableau numpy BGR.")
        if img_bgr.dtype != np.uint8:
            raise TypeError("L'image BGR doit être de type uint8.")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("L'image doit avoir trois dimensions et trois canaux BGR.")
        if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
            raise ValueError("L'image BGR ne peut pas être vide.")

        min_det_score = self._config.detection.min_det_score
        min_face_size = self._config.detection.min_face_size
        faces_with_areas: list[tuple[float, DetectedFace]] = []

        for index, face in enumerate(self._analysis.get(img_bgr)):
            bbox = np.asarray(face.bbox)
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                raise ValueError(
                    f"InsightFace a retourné une boîte invalide pour le visage {index}."
                )
            det_score = float(face.det_score)
            if not np.isfinite(det_score):
                raise ValueError(
                    f"InsightFace a retourné un score invalide pour le visage {index}."
                )

            x1, y1, x2, y2 = (float(coordinate) for coordinate in bbox)
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"InsightFace a retourné une boîte invalide pour le visage {index}."
                )
            if det_score < min_det_score or min(width, height) < min_face_size:
                continue

            faces_with_areas.append((width * height, face))

        faces_with_areas.sort(key=lambda result: result[0], reverse=True)
        return [face for _, face in faces_with_areas]

    def align(
        self,
        img_bgr: NDArray[np.uint8],
        landmarks: NDArray[np.float32],
    ) -> NDArray[np.uint8]:
        """Aligne un visage et retourne un crop BGR de taille 112 × 112.

        Args:
            img_bgr: Image couleur BGR de type uint8.
            landmarks: Cinq points faciaux (yeux, nez, et coins de la bouche),
                de forme `(5, 2)`, tels que retournés par InsightFace.

        Returns:
            Crop facial aligné de forme `(112, 112, 3)` et de type uint8.

        Raises:
            TypeError: Si l'image ou les landmarks n'ont pas le type attendu.
            ValueError: Si l'image, les landmarks ou le crop sont invalides.
            ImportError: Si InsightFace n'est pas installé.
        """
        if not isinstance(img_bgr, np.ndarray) or img_bgr.dtype != np.uint8:
            raise TypeError("L'image BGR doit être un tableau numpy de type uint8.")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("L'image doit avoir trois dimensions et trois canaux BGR.")
        if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
            raise ValueError("L'image BGR ne peut pas être vide.")
        if not isinstance(landmarks, np.ndarray) or not np.issubdtype(landmarks.dtype, np.number):
            raise TypeError("Les landmarks doivent être un tableau numpy numérique.")
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            raise ValueError("Les landmarks doivent contenir cinq points finis de forme (5, 2).")

        try:
            from insightface.utils.face_align import norm_crop
        except ImportError as error:
            raise ImportError(
                "InsightFace est requis pour aligner les visages. "
                "Installez les dépendances depuis requirements.txt."
            ) from error

        aligned = norm_crop(img_bgr, landmark=landmarks, image_size=112)
        if (
            not isinstance(aligned, np.ndarray)
            or aligned.shape != (112, 112, 3)
            or aligned.dtype != np.uint8
        ):
            raise ValueError("InsightFace a retourné un crop aligné invalide.")
        return aligned


def select_main_face(faces: list[DetectedFace]) -> DetectedFace:
    """Sélectionne le visage principal selon l'aire multipliée par son score.

    Args:
        faces: Visages détectés, par exemple le résultat de `FaceDetector.detect`.

    Returns:
        Visage avec la valeur aire × score de détection la plus élevée.

    Raises:
        ValueError: Si aucun visage n'est fourni.
    """
    if not faces:
        raise ValueError("Impossible de sélectionner un visage principal : aucun visage détecté.")

    def face_weight(face: DetectedFace) -> float:
        x1, y1, x2, y2 = (float(coordinate) for coordinate in face.bbox)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return area * float(face.det_score)

    return max(faces, key=face_weight)
