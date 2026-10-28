"""Tests de détection et de sélection du visage principal."""

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import src.detection as detection_module
from src.config import AppConfig, ModelConfig, load_config
from src.detection import DetectedFace, FaceAnalysisBackend, FaceDetector, select_main_face


class FakeFaceAnalysis:
    """Faux moteur permettant de tester les filtres sans charger de modèle."""

    def __init__(self, faces: list[DetectedFace]) -> None:
        self.faces = faces
        self.images_received: list[NDArray[np.uint8]] = []
        self.prepared_with: tuple[int, tuple[int, int]] | None = None

    def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
        """Imite la préparation du moteur InsightFace."""
        self.prepared_with = (ctx_id, det_size)

    def get(self, img_bgr: NDArray[np.uint8]) -> list[DetectedFace]:
        """Retourne les visages configurés pour le test."""
        self.images_received.append(img_bgr)
        return self.faces


def make_face(
    bbox: tuple[float, float, float, float],
    score: float,
) -> DetectedFace:
    """Crée un visage factice compatible avec les attributs InsightFace utilisés."""
    return cast(
        DetectedFace,
        SimpleNamespace(bbox=np.asarray(bbox, dtype=np.float32), det_score=score),
    )


def make_config(min_det_score: float = 0.5, min_face_size: int = 40) -> AppConfig:
    """Retourne la configuration projet avec des seuils adaptés au test."""
    config = load_config()
    return replace(
        config,
        detection=replace(
            config.detection,
            min_det_score=min_det_score,
            min_face_size=min_face_size,
        ),
    )


def test_face_detector_initializes_analysis_once_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le moteur est créé une seule fois puis réutilisé pour chaque image."""
    config = make_config()
    analysis = FakeFaceAnalysis([])
    model_configs: list[ModelConfig] = []

    def create_analysis(model_config: ModelConfig) -> FaceAnalysisBackend:
        model_configs.append(model_config)
        return analysis

    monkeypatch.setattr(detection_module, "_create_analysis", create_analysis)
    detector = FaceDetector(config)
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    detector.detect(image)
    detector.detect(image)

    assert model_configs == [config.model]
    assert analysis.images_received == [image, image]


def test_create_analysis_uses_configured_insightface_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """InsightFace reçoit le pack et les paramètres de modèle configurés."""
    calls: dict[str, object] = {}

    class FakeInsightFace:
        def __init__(self, *, name: str, providers: list[str]) -> None:
            calls["constructor"] = (name, providers)

        def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
            calls["prepare"] = (ctx_id, det_size)

    insightface_module = ModuleType("insightface")
    app_module = ModuleType("insightface.app")
    app_module.FaceAnalysis = FakeInsightFace
    insightface_module.app = app_module
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)

    config = make_config()
    analysis = detection_module._create_analysis(config.model)

    assert isinstance(analysis, FakeInsightFace)
    assert calls["constructor"] == (config.model.name, list(config.model.providers))
    assert calls["prepare"] == (-1, config.model.det_size)


def test_detect_filters_by_score_and_minimum_dimension_then_sorts_by_area(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le seuil et la petite dimension filtrent avant le tri par aire."""
    too_small = make_face((0, 0, 30, 100), 0.99)
    too_weak = make_face((0, 0, 200, 200), 0.49)
    medium = make_face((0, 0, 80, 80), 0.8)
    large = make_face((0, 0, 100, 90), 0.7)
    analysis = FakeFaceAnalysis([medium, too_small, large, too_weak])
    monkeypatch.setattr(detection_module, "_create_analysis", lambda _: analysis)
    detector = FaceDetector(make_config(min_det_score=0.5, min_face_size=40))

    faces = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))

    assert len(faces) == 2
    assert faces[0] is large
    assert faces[1] is medium


def test_detect_includes_faces_at_exact_score_and_size_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Les scores et dimensions égaux aux seuils configurés sont conservés."""
    at_threshold = make_face((0, 0, 40, 60), 0.5)
    analysis = FakeFaceAnalysis([at_threshold])
    monkeypatch.setattr(detection_module, "_create_analysis", lambda _: analysis)
    detector = FaceDetector(make_config(min_det_score=0.5, min_face_size=40))

    faces = detector.detect(np.zeros((80, 80, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0] is at_threshold


@pytest.mark.parametrize(
    ("image", "error_type", "message"),
    [
        (None, TypeError, "tableau numpy"),
        (np.zeros((20, 20, 3), dtype=np.float32), TypeError, "uint8"),
        (np.zeros((20, 20), dtype=np.uint8), ValueError, "trois dimensions"),
        (np.zeros((0, 20, 3), dtype=np.uint8), ValueError, "ne peut pas être vide"),
    ],
)
def test_detect_rejects_invalid_images(
    monkeypatch: pytest.MonkeyPatch,
    image: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Les images incompatibles sont signalées avant l'appel au modèle."""
    analysis = FakeFaceAnalysis([])
    monkeypatch.setattr(detection_module, "_create_analysis", lambda _: analysis)
    detector = FaceDetector(make_config())

    with pytest.raises(error_type, match=message):
        detector.detect(cast(NDArray[np.uint8], image))

    assert analysis.images_received == []


def test_detect_rejects_invalid_model_bounding_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Une boîte mal formée du moteur ne devient pas un résultat silencieux."""
    invalid_face = cast(
        DetectedFace,
        SimpleNamespace(bbox=np.asarray([0, 0, 10], dtype=np.float32), det_score=0.9),
    )
    analysis = FakeFaceAnalysis([invalid_face])
    monkeypatch.setattr(detection_module, "_create_analysis", lambda _: analysis)
    detector = FaceDetector(make_config())

    with pytest.raises(ValueError, match="boîte invalide"):
        detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))


def test_align_returns_crop_with_expected_shape_and_uses_five_landmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L'alignement appelle norm_crop et retourne un crop 112 × 112 × 3."""
    calls: dict[str, object] = {}
    aligned_crop = np.zeros((112, 112, 3), dtype=np.uint8)

    def norm_crop(
        image: NDArray[np.uint8],
        *,
        landmark: NDArray[np.float32],
        image_size: int,
    ) -> NDArray[np.uint8]:
        calls["arguments"] = (image, landmark, image_size)
        return aligned_crop

    insightface_module = ModuleType("insightface")
    utils_module = ModuleType("insightface.utils")
    face_align_module = ModuleType("insightface.utils.face_align")
    face_align_module.norm_crop = norm_crop
    insightface_module.utils = utils_module
    utils_module.face_align = face_align_module
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.utils", utils_module)
    monkeypatch.setitem(sys.modules, "insightface.utils.face_align", face_align_module)
    monkeypatch.setattr(detection_module, "_create_analysis", lambda _: FakeFaceAnalysis([]))
    detector = FaceDetector(make_config())
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    landmarks = np.asarray(
        [[100, 80], [140, 80], [120, 110], [105, 140], [135, 140]],
        dtype=np.float32,
    )

    result = detector.align(image, landmarks)

    assert result.shape == (112, 112, 3)
    assert result.dtype == np.uint8
    assert result is aligned_crop
    arguments = calls["arguments"]
    assert isinstance(arguments, tuple)
    assert arguments[0] is image
    assert arguments[1] is landmarks
    assert arguments[2] == 112


@pytest.mark.parametrize(
    ("landmarks", "error_type", "message"),
    [
        (np.zeros((4, 2), dtype=np.float32), ValueError, "cinq points"),
        (np.full((5, 2), np.nan, dtype=np.float32), ValueError, "cinq points finis"),
        (np.zeros((5, 2), dtype=object), TypeError, "tableau numpy numérique"),
    ],
)
def test_align_rejects_invalid_landmarks(
    monkeypatch: pytest.MonkeyPatch,
    landmarks: NDArray[object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Les landmarks mal formés sont refusés avant le chargement du modèle."""
    monkeypatch.setattr(detection_module, "_create_analysis", lambda _: FakeFaceAnalysis([]))
    detector = FaceDetector(make_config())

    with pytest.raises(error_type, match=message):
        detector.align(np.zeros((100, 100, 3), dtype=np.uint8), landmarks)


def test_select_main_face_uses_area_times_detection_score() -> None:
    """Le visage principal est celui dont l'aire pondérée par le score est maximale."""
    large_moderate_score = make_face((0, 0, 100, 100), 0.7)
    smaller_high_score = make_face((0, 0, 80, 80), 0.9)

    assert select_main_face([smaller_high_score, large_moderate_score]) is large_moderate_score


def test_select_main_face_rejects_empty_list() -> None:
    """Une liste vide produit une erreur compréhensible."""
    with pytest.raises(ValueError, match="aucun visage détecté"):
        select_main_face([])
