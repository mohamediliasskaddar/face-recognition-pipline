"""Tests d'extraction et de normalisation des embeddings."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

import src.embedding as embedding_module
from src.config import AppConfig, ModelConfig, load_config
from src.detection import DetectedFace, FaceAnalysisBackend, FaceDetector, RecognitionModel
from src.embedding import EMBEDDING_DIMENSION, FaceEmbedder, InsightFaceEmbedder


class FakeRecognitionModel:
    """Faux ArcFace qui garde les crops reçus et retourne un vecteur configurable."""

    def __init__(self, output: NDArray[np.float32]) -> None:
        self.output = output
        self.inputs: list[NDArray[np.uint8]] = []

    def get_feat(self, imgs: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Retourne la sortie configurée pour ce test."""
        self.inputs.append(imgs)
        return self.output


class FakeFaceAnalysis:
    """Moteur minimal partagé par le détecteur et l'embedder."""

    def __init__(self, recognition_model: RecognitionModel) -> None:
        self.models: dict[str, RecognitionModel] = {"recognition": recognition_model}

    def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
        """Imite la préparation de FaceAnalysis."""

    def get(self, img_bgr: NDArray[np.uint8]) -> list[DetectedFace]:
        """Imite l'API de détection FaceAnalysis."""
        return []


def make_config() -> AppConfig:
    """Retourne la configuration projet pour les tests."""
    return load_config()


def make_embedder(
    monkeypatch: pytest.MonkeyPatch,
    output: NDArray[np.float32],
) -> tuple[InsightFaceEmbedder, FakeRecognitionModel]:
    """Construit l'embedder avec un modèle ArcFace simulé."""
    recognition_model = FakeRecognitionModel(output)
    analysis = FakeFaceAnalysis(recognition_model)
    monkeypatch.setattr(embedding_module, "_create_analysis", lambda _: analysis)

    return InsightFaceEmbedder(make_config()), recognition_model


def test_insightface_embedder_returns_normalized_float32_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L'embedding retourné est un vecteur 512-d float32 de norme L2 unitaire."""
    raw_embedding = np.arange(1, EMBEDDING_DIMENSION + 1, dtype=np.float32)[None, :]
    embedder, recognition_model = make_embedder(monkeypatch, raw_embedding)
    aligned_face = np.zeros((112, 112, 3), dtype=np.uint8)

    embedding = embedder.embed(aligned_face)

    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-6)
    assert recognition_model.inputs == [aligned_face]


def test_insightface_embedder_reuses_embedding_from_detected_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un embedding FaceAnalysis existant évite de recalculer ArcFace sur le crop."""
    embedder, recognition_model = make_embedder(
        monkeypatch,
        np.ones((1, EMBEDDING_DIMENSION), dtype=np.float32),
    )
    detected_embedding = np.arange(1, EMBEDDING_DIMENSION + 1, dtype=np.float32)
    face = SimpleNamespace(normed_embedding=detected_embedding)
    aligned_face = np.zeros((112, 112, 3), dtype=np.uint8)

    embedding = embedder.embed_detected_face(face, aligned_face)

    np.testing.assert_allclose(
        embedding,
        detected_embedding / np.linalg.norm(detected_embedding),
        atol=1e-6,
    )
    assert recognition_model.inputs == []


def test_face_embedder_is_an_abstract_interface() -> None:
    """L'interface ne peut pas être instanciée sans implémenter embed."""
    with pytest.raises(TypeError, match="abstract"):
        FaceEmbedder()


def test_embedder_and_detector_can_share_one_face_analysis_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le détecteur et l'embedder peuvent réutiliser le même moteur chargé."""
    recognition_model = FakeRecognitionModel(np.ones((1, EMBEDDING_DIMENSION), dtype=np.float32))
    analysis = FakeFaceAnalysis(recognition_model)
    create_calls: list[ModelConfig] = []

    def create_analysis(model_config: ModelConfig) -> FaceAnalysisBackend:
        create_calls.append(model_config)
        return analysis

    monkeypatch.setattr(embedding_module, "_create_analysis", create_analysis)
    config = make_config()
    embedder = InsightFaceEmbedder(config)
    detector = FaceDetector(config, analysis=embedder.analysis)

    assert detector.analysis is embedder.analysis
    assert create_calls == [config.model]


@pytest.mark.parametrize(
    ("raw_embedding", "error_message"),
    [
        (np.ones((1, 511), dtype=np.float32), "dimension 512"),
        (np.full((1, 512), np.nan, dtype=np.float32), "non fini"),
        (np.zeros((1, 512), dtype=np.float32), "norme nulle"),
    ],
)
def test_embed_rejects_invalid_model_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    raw_embedding: NDArray[np.float32],
    error_message: str,
) -> None:
    """Les vecteurs ArcFace nuls, non finis ou mal dimensionnés sont refusés."""
    embedder, _ = make_embedder(monkeypatch, raw_embedding)

    with pytest.raises(ValueError, match=error_message):
        embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8))


def test_embed_rejects_unaligned_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L'embedder refuse un crop qui n'est pas de taille 112 × 112."""
    embedder, recognition_model = make_embedder(
        monkeypatch,
        np.ones((1, EMBEDDING_DIMENSION), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="forme"):
        embedder.embed(np.zeros((100, 100, 3), dtype=np.uint8))

    assert recognition_model.inputs == []


def test_embedder_rejects_analysis_without_recognition_model() -> None:
    """Une configuration moteur sans modèle ArcFace donne une erreur explicite."""
    config = make_config()
    analysis = FakeFaceAnalysis(FakeRecognitionModel(np.ones((1, 512), dtype=np.float32)))
    analysis.models.clear()

    with pytest.raises(RuntimeError, match="modèle 'recognition'"):
        InsightFaceEmbedder(config, analysis=analysis)
