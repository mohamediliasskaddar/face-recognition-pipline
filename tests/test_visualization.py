"""Tests de génération des annotations et visualisations du pipeline."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.types import FaceRecord, MatchResult
from src.visualization import draw_faces, make_result_grid, plot_similarity_heatmap


def make_record(
    image_path: Path,
    embedding: list[float],
    *,
    bbox: tuple[int, int, int, int] = (2, 2, 18, 18),
) -> FaceRecord:
    """Construit un visage synthétique pour les tests."""
    return FaceRecord(
        image_path=str(image_path),
        bbox=bbox,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=0.93,
        aligned=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def write_image(path: Path, value: int) -> None:
    """Écrit une petite image BGR synthétique."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((24, 24, 3), value, dtype=np.uint8))


def test_draw_faces_adds_box_and_score_without_mutating_source(tmp_path: Path) -> None:
    """L'annotation retourne une copie modifiée avec la couleur attendue."""
    source = np.zeros((24, 24, 3), dtype=np.uint8)
    record = make_record(tmp_path / "face.jpg", [1.0, 0.0])

    annotated = draw_faces(source, [record])

    assert annotated.shape == source.shape
    assert annotated.dtype == np.uint8
    assert np.count_nonzero(annotated) > 0
    assert not np.count_nonzero(source)


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (np.zeros((10, 10), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 3), dtype=np.float32), TypeError),
    ],
)
def test_draw_faces_rejects_invalid_images(image: np.ndarray, error: type[Exception]) -> None:
    """Les images non BGR uint8 sont refusées explicitement."""
    with pytest.raises(error):
        draw_faces(image, [])


def test_make_result_grid_writes_query_and_annotated_matches(tmp_path: Path) -> None:
    """La grille exportée contient une tuile requête et chaque résultat."""
    query = np.full((24, 24, 3), 30, dtype=np.uint8)
    image_a = tmp_path / "gallery" / "a.jpg"
    image_b = tmp_path / "gallery" / "b.jpg"
    write_image(image_a, 100)
    write_image(image_b, 180)
    records = [
        make_record(image_a, [1.0, 0.0]),
        make_record(image_b, [0.0, 1.0]),
    ]
    output = tmp_path / "outputs" / "result_grid.png"

    make_result_grid(
        query,
        [
            MatchResult(str(image_a), 0.9, 0),
            MatchResult(str(image_b), 0.8, 0),
        ],
        records,
        output,
    )

    rendered = cv2.imread(str(output))
    assert rendered is not None
    assert rendered.shape[0] > query.shape[0]
    assert rendered.shape[1] > query.shape[1]


def test_make_result_grid_handles_no_matches(tmp_path: Path) -> None:
    """Sans match, la grille de sortie contient quand même la requête."""
    output = tmp_path / "result_grid.png"

    make_result_grid(np.zeros((24, 24, 3), dtype=np.uint8), [], [], output)

    assert cv2.imread(str(output)) is not None


def test_plot_similarity_heatmap_writes_face_similarity_matrix(tmp_path: Path) -> None:
    """La heatmap est écrite et la comparaison des vecteurs opposés est visible."""
    records = [
        make_record(tmp_path / "a.jpg", [1.0, 0.0]),
        make_record(tmp_path / "b.jpg", [-1.0, 0.0]),
    ]
    output = tmp_path / "heatmap.png"

    plot_similarity_heatmap(records, output)

    image = cv2.imread(str(output))
    assert image is not None
    assert image.shape[0] > 100
    assert image.shape[1] > 100


def test_plot_similarity_heatmap_handles_empty_gallery(tmp_path: Path) -> None:
    """Une galerie vide donne une figure explicative au lieu d'une erreur."""
    output = tmp_path / "empty_heatmap.png"

    plot_similarity_heatmap([], output)

    assert cv2.imread(str(output)) is not None
