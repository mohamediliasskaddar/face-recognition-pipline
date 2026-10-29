"""Tests d'intégration InsightFace sur les fixtures locales du projet."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import src.pipeline as pipeline_module
from src.config import AppConfig, load_config
from src.detection import FaceDetector
from src.embedding import InsightFaceEmbedder
from src.pipeline import PipelineError, run

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MODEL_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


def _read_fixture(filename: str, *, max_dimension: int = 960) -> np.ndarray:
    """Lit une fixture BGR et la réduit pour limiter le coût d'inférence."""
    image_path = FIXTURE_DIR / filename
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError(f"Fixture image illisible : {image_path}")
    scale = min(1.0, max_dimension / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return image


def _model_pack_is_installed(model_name: str) -> bool:
    """Vérifie la présence locale des poids avant que FaceAnalysis ne les télécharge."""
    model_dir = Path.home() / ".insightface" / "models" / model_name
    return all((model_dir / filename).is_file() for filename in MODEL_FILES)


@pytest.fixture(scope="module")
def real_backend() -> tuple[AppConfig, FaceDetector, InsightFaceEmbedder]:
    """Partage les vrais modèles dans ce module et ne déclenche aucun téléchargement."""
    config = load_config()
    if not _model_pack_is_installed(config.model.name):
        pytest.skip(
            f"Poids InsightFace '{config.model.name}' absents localement; "
            "le test réel est ignoré sans téléchargement."
        )

    embedder = InsightFaceEmbedder(config)
    detector = FaceDetector(config, analysis=embedder.analysis)
    return config, detector, embedder


def test_real_detection_alignment_and_embedding_on_face_fixtures(
    real_backend: tuple[AppConfig, FaceDetector, InsightFaceEmbedder],
) -> None:
    """Les fixtures valident l'absence de visage, le multi-visage et les formes."""
    _, detector, embedder = real_backend
    no_face_image = _read_fixture("pic-with-no-face.png")
    composite_image = _read_fixture("three-image-fusionne.jpg")

    assert detector.detect(no_face_image) == []

    faces = detector.detect(composite_image)
    assert len(faces) == 3
    for face in faces:
        aligned = detector.align(composite_image, face.kps)
        embedding = embedder.embed_detected_face(face, aligned)

        assert aligned.shape == (112, 112, 3)
        assert aligned.dtype == np.uint8
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_real_pipeline_handles_four_gallery_images_and_multi_face_query(
    real_backend: tuple[AppConfig, FaceDetector, InsightFaceEmbedder],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Le pipeline réel indexe quatre images et recherche depuis une requête multi-visage."""
    config, detector, embedder = real_backend
    gallery = tmp_path / "gallery"
    gallery.mkdir()

    composite = cv2.imread(str(FIXTURE_DIR / "three-image-fusionne.jpg"), cv2.IMREAD_COLOR)
    assert composite is not None
    panel_width = composite.shape[1] // 3
    for index in range(3):
        panel = composite[:, index * panel_width : (index + 1) * panel_width]
        assert cv2.imwrite(str(gallery / f"person_{index}.jpg"), panel)

    no_face_image = _read_fixture("pic-with-no-face.png", max_dimension=640)
    assert cv2.imwrite(str(gallery / "landscape.png"), no_face_image)
    query = FIXTURE_DIR / "three-image-fusionne.jpg"

    def reuse_embedder(_: AppConfig) -> InsightFaceEmbedder:
        return embedder

    def reuse_detector(
        _: AppConfig,
        *,
        analysis: object,
    ) -> FaceDetector:
        assert analysis is detector.analysis
        return detector

    monkeypatch.setattr(pipeline_module, "InsightFaceEmbedder", reuse_embedder)
    monkeypatch.setattr(pipeline_module, "FaceDetector", reuse_detector)

    matches = run(
        query,
        gallery,
        tmp_path / "output",
        config,
        face_index=0,
        use_cache=False,
    )

    assert [Path(match.image_path).name for match in matches] == ["person_2.jpg"]
    assert matches[0].best_score > 0.9
    assert (tmp_path / "output" / "matches" / "person_2.jpg").is_file()
    assert len(list(gallery.iterdir())) == 4

    no_face_query = tmp_path / "no-face-query.png"
    assert cv2.imwrite(str(no_face_query), no_face_image)
    with pytest.raises(PipelineError, match="Aucun visage détecté"):
        run(
            no_face_query,
            gallery,
            tmp_path / "no-face-output",
            config,
            use_cache=False,
        )
