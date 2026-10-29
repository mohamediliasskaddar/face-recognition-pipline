"""Tests d'intégration du pipeline et de son interface CLI."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import cv2
import numpy as np
import pytest

import src.main as main_module
import src.pipeline as pipeline_module
from src.config import AppConfig, load_config
from src.detection import DetectedFace
from src.pipeline import PipelineError, run
from src.types import FaceRecord, MatchResult


class FakeEmbedder:
    """Faux embedder/analysis permettant d'éviter le chargement du modèle réel."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.analysis = object()

    def embed_detected_face(
        self,
        face: DetectedFace,
        aligned_face: np.ndarray,
    ) -> np.ndarray:
        """Retourne l'embedding placé sur le faux visage."""
        assert aligned_face.shape == (112, 112, 3)
        return face.normed_embedding


class FakeDetector:
    """Faux détecteur qui expose des visages configurables par test."""

    faces: ClassVar[list[DetectedFace]] = []

    def __init__(self, config: AppConfig, *, analysis: object) -> None:
        assert analysis is not None
        self.config = config

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Retourne les visages configurés pour le test."""
        return self.faces

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Retourne un crop aligné synthétique."""
        return np.zeros((112, 112, 3), dtype=np.uint8)


def make_face(embedding: list[float], *, size: int = 20) -> DetectedFace:
    """Construit un faux visage compatible avec les interfaces du pipeline."""
    return cast(
        DetectedFace,
        SimpleNamespace(
            bbox=np.asarray([0, 0, size, size], dtype=np.float32),
            kps=np.zeros((5, 2), dtype=np.float32),
            det_score=0.9,
            normed_embedding=np.asarray(embedding, dtype=np.float32),
        ),
    )


def make_record(image_path: Path, embedding: list[float]) -> FaceRecord:
    """Construit un enregistrement de galerie synthétique."""
    return FaceRecord(
        image_path=str(image_path),
        bbox=(0, 0, 20, 20),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=0.9,
        aligned=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.asarray(embedding, dtype=np.float32),
    )


@pytest.fixture
def pipeline_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, AppConfig, list[FaceRecord]]:
    """Configure un pipeline simulé et retourne ses chemins et index."""
    query = tmp_path / "query.jpg"
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir()
    assert cv2.imwrite(str(query), np.zeros((32, 32, 3), dtype=np.uint8))
    first_image = gallery / "first.jpg"
    second_image = gallery / "second.jpg"
    assert cv2.imwrite(str(first_image), np.full((32, 32, 3), 80, dtype=np.uint8))
    assert cv2.imwrite(str(second_image), np.full((32, 32, 3), 160, dtype=np.uint8))

    config = load_config()
    FakeDetector.faces = [make_face([1.0, 0.0])]
    records = [
        make_record(first_image, [1.0, 0.0]),
        make_record(second_image, [0.0, 1.0]),
    ]
    monkeypatch.setattr(pipeline_module, "InsightFaceEmbedder", FakeEmbedder)
    monkeypatch.setattr(pipeline_module, "FaceDetector", FakeDetector)
    monkeypatch.setattr(
        pipeline_module,
        "read_image",
        lambda _: np.zeros((32, 32, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        pipeline_module,
        "list_images",
        lambda _: [first_image, second_image],
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_gallery_index",
        lambda *args, **kwargs: records,
    )
    return query, gallery, output, config, records


def test_run_exports_matching_images_and_report(
    pipeline_setup: tuple[Path, Path, Path, AppConfig, list[FaceRecord]],
) -> None:
    """Le pipeline copie les correspondances et écrit un rapport JSON."""
    query, gallery, output, config, _ = pipeline_setup

    matches = run(query, gallery, output, config)

    assert matches == [MatchResult(str(gallery / "first.jpg"), 1.0, 0)]
    assert (output / "matches" / "first.jpg").read_bytes() == (gallery / "first.jpg").read_bytes()
    assert (output / "annotated" / "first.jpg").is_file()
    assert (output / "annotated" / "query_query.png").is_file()
    assert (output / "result_grid.png").is_file()
    assert (output / "similarity_heatmap.png").is_file()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["query"] == str(query)
    assert report["threshold"] == config.matching.threshold
    assert report["n_gallery_images"] == 2
    assert report["n_matches"] == 1
    assert report["matches"][0]["image"] == str(gallery / "first.jpg")


def test_run_uses_requested_face_index(
    pipeline_setup: tuple[Path, Path, Path, AppConfig, list[FaceRecord]],
) -> None:
    """L'option face_index choisit bien le visage demandé."""
    query, gallery, output, config, _ = pipeline_setup
    FakeDetector.faces = [
        make_face([0.0, 1.0]),
        make_face([1.0, 0.0], size=10),
    ]

    matches = run(query, gallery, output, config, face_index=1)

    assert matches[0].image_path == str(gallery / "first.jpg")
    assert matches[0].best_score == pytest.approx(1.0)


def test_run_rejects_missing_face_and_invalid_face_index(
    pipeline_setup: tuple[Path, Path, Path, AppConfig, list[FaceRecord]],
) -> None:
    """Une requête sans visage ou un index absent produit une erreur claire."""
    query, gallery, output, config, _ = pipeline_setup
    FakeDetector.faces = []
    with pytest.raises(PipelineError, match="Aucun visage détecté"):
        run(query, gallery, output, config)

    FakeDetector.faces = [make_face([1.0, 0.0])]
    with pytest.raises(PipelineError, match="index de visage 2 est invalide"):
        run(query, gallery, output, config, face_index=2)


def test_run_rejects_empty_gallery_and_empty_face_index(
    pipeline_setup: tuple[Path, Path, Path, AppConfig, list[FaceRecord]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le pipeline distingue un dossier vide d'une galerie sans visage indexable."""
    query, gallery, output, config, _ = pipeline_setup
    monkeypatch.setattr(pipeline_module, "list_images", lambda _: [])
    with pytest.raises(PipelineError, match="Aucune image trouvée"):
        run(query, gallery, output, config)

    monkeypatch.setattr(pipeline_module, "list_images", lambda _: [gallery / "first.jpg"])
    monkeypatch.setattr(pipeline_module, "build_gallery_index", lambda *args, **kwargs: [])
    with pytest.raises(PipelineError, match="Aucun visage exploitable"):
        run(query, gallery, output, config)


def test_run_exports_report_when_no_image_matches(
    pipeline_setup: tuple[Path, Path, Path, AppConfig, list[FaceRecord]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aucun match est un résultat valide : la commande laisse un rapport vide."""
    query, gallery, output, config, records = pipeline_setup
    monkeypatch.setattr(
        pipeline_module,
        "build_gallery_index",
        lambda *args, **kwargs: [
            make_record(Path(records[0].image_path), [0.0, 1.0]),
            make_record(Path(records[1].image_path), [-1.0, 0.0]),
        ],
    )

    assert run(query, gallery, output, config) == []
    assert (output / "matches").is_dir()
    assert not list((output / "matches").iterdir())
    assert (output / "result_grid.png").is_file()
    assert (output / "similarity_heatmap.png").is_file()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["n_matches"] == 0
    assert report["matches"] == []


def test_run_rejects_unimplemented_backend(
    pipeline_setup: tuple[Path, Path, Path, AppConfig, list[FaceRecord]],
) -> None:
    """Un backend non pris en charge n'est pas remplacé silencieusement."""
    query, gallery, output, config, _ = pipeline_setup
    unsupported_config = config.__class__(
        model=config.model.__class__(
            backend="facenet",
            name=config.model.name,
            det_size=config.model.det_size,
            providers=config.model.providers,
        ),
        detection=config.detection,
        matching=config.matching,
        paths=config.paths,
    )

    with pytest.raises(PipelineError, match="n'est pas implémenté"):
        run(query, gallery, output, unsupported_config)


def test_cli_applies_options_and_returns_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Le CLI transmet seuil, backend, index de visage et cache au pipeline."""
    config = load_config()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(main_module, "load_config", lambda: config)
    monkeypatch.setattr(
        main_module,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or [],
    )

    status = main_module.main(
        [
            "--query",
            str(tmp_path / "query.jpg"),
            "--gallery",
            str(tmp_path / "gallery"),
            "--output",
            str(tmp_path / "output"),
            "--threshold",
            "0.55",
            "--face-index",
            "1",
            "--backend",
            "insightface",
            "--no-cache",
        ]
    )

    assert status == 0
    assert calls[0][0][3].matching.threshold == 0.55
    assert calls[0][1] == {"face_index": 1, "use_cache": False}


def test_cli_returns_nonzero_when_pipeline_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Une erreur métier donne un code de sortie non nul."""
    monkeypatch.setattr(main_module, "load_config", load_config)

    def fail_run(*args: object, **kwargs: object) -> list[MatchResult]:
        raise PipelineError("aucun visage")

    monkeypatch.setattr(main_module, "run", fail_run)
    status = main_module.main(
        [
            "--query",
            str(tmp_path / "query.jpg"),
            "--gallery",
            str(tmp_path / "gallery"),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert status == 1
