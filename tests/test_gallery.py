"""Tests de l'indexation multi-visages et du cache de galerie."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import AppConfig, load_config
from src.detection import DetectedFace, FaceDetector
from src.embedding import FaceEmbedder
from src.gallery import GalleryCacheError, build_gallery_index, load_cache, save_cache


class FakeDetector:
    """Détecteur factice dont les résultats dépendent du marqueur de l'image."""

    def __init__(self) -> None:
        self.config = load_config()
        self.detect_calls: list[int] = []
        self.align_calls = 0

    def detect(self, img_bgr: NDArray[np.uint8]) -> list[DetectedFace]:
        marker = int(img_bgr[0, 0, 0])
        self.detect_calls.append(marker)
        if marker == 2:
            return []
        face_count = 2 if marker == 1 else 1
        faces = []
        for index in range(face_count):
            x_offset = index * 10
            faces.append(
                cast(
                    DetectedFace,
                    SimpleNamespace(
                        bbox=np.asarray(
                            [x_offset, 0, x_offset + 20, 25],
                            dtype=np.float32,
                        ),
                        kps=np.full((5, 2), marker + index, dtype=np.float32),
                        det_score=0.9 - index * 0.1,
                    ),
                )
            )
        return faces

    def align(
        self,
        img_bgr: NDArray[np.uint8],
        landmarks: NDArray[np.float32],
    ) -> NDArray[np.uint8]:
        assert img_bgr.ndim == 3
        self.align_calls += 1
        return np.full((112, 112, 3), round(float(landmarks[0, 0])), dtype=np.uint8)


class FakeEmbedder(FaceEmbedder):
    """Embedder factice qui retourne un vecteur unité identifiable par pixel."""

    def __init__(self) -> None:
        self.config: AppConfig = load_config()
        self.embed_calls = 0

    def embed(self, aligned_face: NDArray[np.uint8]) -> NDArray[np.float32]:
        self.embed_calls += 1
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        embedding[1] = aligned_face[0, 0, 0] / 10.0
        return embedding / np.linalg.norm(embedding)


def write_marker_image(path: Path, marker: int) -> None:
    """Écrit une petite image PNG avec un marqueur BGR pour le faux détecteur."""
    image = np.full((32, 32, 3), marker, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_build_gallery_index_keeps_every_face_and_skips_empty_or_corrupt_images(
    tmp_path: Path,
) -> None:
    """Chaque visage est enregistré, sans visage et fichiers corrompus sont ignorés."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    write_marker_image(gallery / "01_two_faces.png", 1)
    write_marker_image(gallery / "02_no_face.png", 2)
    write_marker_image(gallery / "03_one_face.png", 3)
    (gallery / "04_corrupt.jpg").write_text("not an image", encoding="utf-8")
    detector = FakeDetector()
    embedder = FakeEmbedder()

    records = build_gallery_index(
        gallery,
        cast(FaceDetector, detector),
        embedder,
        cache_path=tmp_path / "cache" / "embeddings.npz",
    )

    assert len(records) == 3
    assert [Path(record.image_path).name for record in records] == [
        "01_two_faces.png",
        "01_two_faces.png",
        "03_one_face.png",
    ]
    assert records[0].bbox == (0, 0, 20, 25)
    assert records[1].bbox == (10, 0, 30, 25)
    assert records[0].landmarks.shape == (5, 2)
    assert records[0].aligned.shape == (112, 112, 3)
    assert records[0].embedding.shape == (512,)
    assert len(detector.detect_calls) == 3
    assert detector.align_calls == 3
    assert embedder.embed_calls == 3


def test_build_gallery_index_reuses_cache_without_reprocessing_images(
    tmp_path: Path,
) -> None:
    """Un cache valide évite la détection, l'alignement et l'embedding suivants."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    write_marker_image(gallery / "one.png", 1)
    cache_path = tmp_path / "cache" / "embeddings.npz"
    detector = FakeDetector()
    embedder = FakeEmbedder()

    first_records = build_gallery_index(
        gallery,
        cast(FaceDetector, detector),
        embedder,
        cache_path=cache_path,
    )
    second_records = build_gallery_index(
        gallery,
        cast(FaceDetector, detector),
        embedder,
        cache_path=cache_path,
    )

    assert len(first_records) == 2
    assert len(second_records) == 2
    assert [record.image_path for record in second_records] == [
        record.image_path for record in first_records
    ]
    assert np.array_equal(second_records[0].embedding, first_records[0].embedding)
    assert detector.detect_calls == [1]
    assert detector.align_calls == 2
    assert embedder.embed_calls == 2


@pytest.mark.parametrize("change", ["contents", "file_list", "mtime"])
def test_build_gallery_index_invalidates_cache_when_gallery_changes(
    tmp_path: Path,
    change: str,
) -> None:
    """Un changement de contenu ou de liste des images force le recalcul."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    image_path = gallery / "one.png"
    write_marker_image(image_path, 1)
    cache_path = tmp_path / "cache" / "embeddings.npz"
    detector = FakeDetector()
    embedder = FakeEmbedder()
    build_gallery_index(
        gallery,
        cast(FaceDetector, detector),
        embedder,
        cache_path=cache_path,
    )

    if change == "contents":
        write_marker_image(image_path, 4)
    elif change == "file_list":
        write_marker_image(gallery / "two.png", 3)
    else:
        stat = image_path.stat()
        os.utime(image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    build_gallery_index(
        gallery,
        cast(FaceDetector, detector),
        embedder,
        cache_path=cache_path,
    )

    expected_calls = 3 if change == "file_list" else 2
    assert len(detector.detect_calls) == expected_calls


def test_load_cache_returns_none_when_model_configuration_changes(
    tmp_path: Path,
) -> None:
    """Un cache associé à une autre configuration est considéré périmé."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    write_marker_image(gallery / "one.png", 1)
    cache_path = tmp_path / "cache" / "embeddings.npz"
    records = build_gallery_index(
        gallery,
        cast(FaceDetector, FakeDetector()),
        FakeEmbedder(),
        cache_path=cache_path,
    )

    assert records
    assert (
        load_cache(
            cache_path,
            gallery_fingerprint="different-gallery",
            configuration_fingerprint="different-config",
        )
        is None
    )


def test_build_gallery_index_rejects_corrupt_existing_cache(tmp_path: Path) -> None:
    """Un cache présent mais corrompu signale l'erreur au lieu d'être caché."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    write_marker_image(gallery / "one.png", 1)
    cache_path = tmp_path / "cache" / "embeddings.npz"
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"corrupt cache")

    with pytest.raises(GalleryCacheError, match="Impossible de lire le cache"):
        build_gallery_index(
            gallery,
            cast(FaceDetector, FakeDetector()),
            FakeEmbedder(),
            cache_path=cache_path,
        )


def test_empty_gallery_is_cached_as_an_empty_index(tmp_path: Path) -> None:
    """Une galerie sans visage produit et recharge un cache vide valide."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    write_marker_image(gallery / "no_face.png", 2)
    cache_path = tmp_path / "cache" / "embeddings.npz"
    detector = FakeDetector()
    embedder = FakeEmbedder()

    first_records = build_gallery_index(
        gallery,
        cast(FaceDetector, detector),
        embedder,
        cache_path=cache_path,
    )
    second_records = build_gallery_index(
        gallery,
        cast(FaceDetector, detector),
        embedder,
        cache_path=cache_path,
    )

    assert first_records == []
    assert second_records == []
    assert detector.detect_calls == [2]


def test_cache_round_trip_preserves_face_record_values(tmp_path: Path) -> None:
    """Le cache NPZ restaure les tableaux et métadonnées des FaceRecord."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    write_marker_image(gallery / "one.png", 3)
    cache_path = tmp_path / "cache" / "embeddings.npz"
    records = build_gallery_index(
        gallery,
        cast(FaceDetector, FakeDetector()),
        FakeEmbedder(),
        cache_path=cache_path,
    )
    config_fingerprint = "config"
    gallery_fingerprint = "gallery"
    save_cache(
        cache_path,
        records,
        gallery_fingerprint=gallery_fingerprint,
        configuration_fingerprint=config_fingerprint,
    )
    loaded_records = load_cache(
        cache_path,
        gallery_fingerprint=gallery_fingerprint,
        configuration_fingerprint=config_fingerprint,
    )

    assert loaded_records is not None
    assert len(loaded_records) == 1
    assert loaded_records[0].image_path == records[0].image_path
    assert loaded_records[0].bbox == records[0].bbox
    assert loaded_records[0].det_score == pytest.approx(records[0].det_score)
    np.testing.assert_array_equal(loaded_records[0].landmarks, records[0].landmarks)
    np.testing.assert_array_equal(loaded_records[0].aligned, records[0].aligned)
    np.testing.assert_array_equal(loaded_records[0].embedding, records[0].embedding)
