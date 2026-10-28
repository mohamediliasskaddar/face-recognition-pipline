"""Extraction et normalisation des embeddings faciaux."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.config import AppConfig, load_config
from src.detection import (
    DetectedFace,
    FaceAnalysisBackend,
    RecognitionModel,
    _create_analysis,
)

EMBEDDING_DIMENSION = 512
ALIGNED_FACE_SHAPE = (112, 112, 3)


class FaceEmbedder(ABC):
    """Interface commune des extracteurs d'embeddings faciaux."""

    @abstractmethod
    def embed(self, aligned_face: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Transforme un crop facial aligné en embedding normalisé."""

    def embed_detected_face(
        self,
        face: DetectedFace,
        aligned_face: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Extrait l'embedding d'un visage détecté; le backend peut réutiliser sa sortie."""
        return self.embed(aligned_face)


class InsightFaceEmbedder(FaceEmbedder):
    """Extrait les embeddings ArcFace depuis les crops alignés InsightFace."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        analysis: FaceAnalysisBackend | None = None,
    ) -> None:
        """Charge ou réutilise le moteur InsightFace et son modèle ArcFace.

        Args:
            config: Configuration du pipeline. La configuration du projet est
                chargée si elle n'est pas fournie.
            analysis: Moteur déjà chargé à partager avec `FaceDetector`.

        Raises:
            RuntimeError: Si le moteur ne contient pas de modèle de reconnaissance
                ArcFace utilisable.
        """
        self._config = config if config is not None else load_config()
        self._analysis = analysis if analysis is not None else _create_analysis(self._config.model)
        recognition_model = self._analysis.models.get("recognition")
        if recognition_model is None or not callable(getattr(recognition_model, "get_feat", None)):
            raise RuntimeError(
                "Le moteur InsightFace ne contient pas de modèle 'recognition' compatible."
            )
        self._recognition_model: RecognitionModel = recognition_model

    @property
    def analysis(self) -> FaceAnalysisBackend:
        """Retourne le moteur pour le partager avec le détecteur."""
        return self._analysis

    @property
    def config(self) -> AppConfig:
        """Retourne la configuration utilisée pour l'extraction."""
        return self._config

    def embed(self, aligned_face: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Calcule puis normalise l'embedding L2 d'un crop aligné.

        Args:
            aligned_face: Crop facial BGR de type uint8 et de forme
                `(112, 112, 3)`, produit par `FaceDetector.align`.

        Returns:
            Embedding ArcFace de forme `(512,)`, type float32 et norme L2 égale à 1.

        Raises:
            TypeError: Si l'entrée n'est pas un tableau numpy uint8.
            ValueError: Si le crop, le vecteur brut ou sa norme est invalide.
        """
        if not isinstance(aligned_face, np.ndarray) or aligned_face.dtype != np.uint8:
            raise TypeError("Le visage aligné doit être un tableau numpy de type uint8.")
        if aligned_face.shape != ALIGNED_FACE_SHAPE:
            raise ValueError(
                "Le visage aligné doit avoir la forme "
                f"{ALIGNED_FACE_SHAPE}, forme reçue : {aligned_face.shape}."
            )

        raw_embedding = np.asarray(self._recognition_model.get_feat(aligned_face))
        if raw_embedding.shape == (1, EMBEDDING_DIMENSION):
            raw_embedding = raw_embedding[0]
        if raw_embedding.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                "Le modèle ArcFace doit retourner un embedding de dimension "
                f"{EMBEDDING_DIMENSION}, forme reçue : {raw_embedding.shape}."
            )

        return self._normalize_embedding(raw_embedding)

    def embed_detected_face(
        self,
        face: DetectedFace,
        aligned_face: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Réutilise l'embedding calculé par FaceAnalysis, sans inférence ArcFace en double."""
        normed_embedding = getattr(face, "normed_embedding", None)
        if normed_embedding is None:
            return self.embed(aligned_face)
        return self._normalize_embedding(np.asarray(normed_embedding))

    @staticmethod
    def _normalize_embedding(raw_embedding: NDArray[np.generic]) -> NDArray[np.float32]:
        """Valide et normalise un vecteur de reconnaissance ArcFace."""
        if raw_embedding.shape == (1, EMBEDDING_DIMENSION):
            raw_embedding = raw_embedding[0]
        if raw_embedding.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                "Le modèle ArcFace doit retourner un embedding de dimension "
                f"{EMBEDDING_DIMENSION}, forme reçue : {raw_embedding.shape}."
            )

        embedding = np.asarray(raw_embedding, dtype=np.float32)
        if not np.isfinite(embedding).all():
            raise ValueError("Le modèle ArcFace a retourné un embedding non fini.")

        norm = float(np.linalg.norm(embedding.astype(np.float64)))
        if norm == 0.0:
            raise ValueError("Impossible de normaliser un embedding de norme nulle.")

        return np.asarray(embedding / norm, dtype=np.float32)
