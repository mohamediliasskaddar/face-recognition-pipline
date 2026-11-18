"""Tests de l'export des visages dans des dossiers par personne."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.organizer import organize
from src.types import FaceRecord


def write_source_image(path: Path, color: tuple[int, int, int] = (40, 80, 120)) -> None:
    """Crée une petite image BGR déterministe pour les tests d'export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 40, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def make_record(
    image_path: Path,
    bbox: tuple[int, int, int, int],
) -> FaceRecord:
    """Construit un enregistrement de visage pour une image source existante."""
    return FaceRecord(
        image_path=str(image_path),
        bbox=bbox,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=0.9,
        aligned=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.zeros(512, dtype=np.float32),
    )


def test_organize_numbers_people_by_size_and_exports_unknown_faces(
    tmp_path: Path,
) -> None:
    """Les clusters sont triés par taille et chaque crop garde sa source."""
    gallery = tmp_path / "gallery"
    first_image = gallery / "source_a.png"
    second_image = gallery / "source_b.png"
    unknown_image = gallery / "source_c.png"
    for image_path in (first_image, second_image, unknown_image):
        write_source_image(image_path)
    records = [
        make_record(first_image, (2, 2, 12, 12)),
        make_record(first_image, (20, 20, 30, 30)),
        make_record(second_image, (5, 5, 20, 20)),
        make_record(unknown_image, (5, 5, 20, 20)),
    ]

    report_path = organize(
        records,
        np.asarray([5, 5, 7, -1], dtype=np.int32),
        tmp_path / "output",
        copy_originals=True,
    )
    output = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["n_persons"] == 2
    assert report["n_faces"] == 4
    assert report["folders"]["person_01"]["label"] == 5
    assert report["folders"]["person_01"]["face_count"] == 2
    assert [entry["filename"] for entry in report["folders"]["person_01"]["crops"]] == [
        "source_a_face0.jpg",
        "source_a_face1.jpg",
    ]
    assert report["folders"]["person_02"]["label"] == 7
    assert report["folders"]["unknown"]["face_count"] == 1
    assert (output / "person_01" / "source_a.png").is_file()
    assert (output / "person_02" / "source_b.png").is_file()
    assert not (output / "unknown" / "source_c.png").exists()
    assert cv2.imread(str(output / "person_01" / "source_a_face0.jpg")) is not None


def test_organize_requires_explicit_overwrite_and_replaces_output(
    tmp_path: Path,
) -> None:
    """Un dossier non vide est protégé par défaut et remplacé avec overwrite."""
    source = tmp_path / "gallery" / "source.png"
    write_source_image(source)
    records = [make_record(source, (5, 5, 20, 20))]
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="n'est pas vide"):
        organize(records, [0], output)
    assert marker.is_file()

    report_path = organize(records, [0], output, overwrite=True)

    assert report_path.is_file()
    assert not marker.exists()


def test_organize_rejects_output_inside_source_gallery(
    tmp_path: Path,
) -> None:
    """L'option overwrite ne peut pas effacer le dossier des images source."""
    gallery = tmp_path / "gallery"
    source = gallery / "source.png"
    write_source_image(source)

    with pytest.raises(ValueError, match="images source"):
        organize([make_record(source, (5, 5, 20, 20))], [0], gallery, overwrite=True)

    assert source.is_file()


def test_organize_creates_unknown_folder_for_empty_input(tmp_path: Path) -> None:
    """Une entrée vide produit un rapport valide et le dossier unknown."""
    report_path = organize([], [], tmp_path / "empty_output")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["n_persons"] == 0
    assert report["n_faces"] == 0
    assert report["folders"]["unknown"]["crops"] == []
