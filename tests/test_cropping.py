"""Tests du recadrage des visages."""

import numpy as np
import pytest

from src.cropping import crop_face


def test_crop_face_adds_relative_margin_and_returns_square_crop() -> None:
    """La marge s'ajoute autour de la boîte et le crop intérieur reste carré."""
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)

    crop = crop_face(image, (8, 6, 12, 10), margin=0.25)

    assert crop.shape == (6, 6, 3)
    np.testing.assert_array_equal(crop, image[5:11, 7:13])
    assert crop.flags.owndata


@pytest.mark.parametrize(
    ("bbox", "expected_shape", "expected_slice"),
    [
        ((0, 0, 4, 4), (8, 8, 3), (slice(0, 8), slice(0, 8))),
        ((8, 6, 12, 10), (8, 8, 3), (slice(2, 10), slice(4, 12))),
    ],
)
def test_crop_face_clips_at_image_edges(
    bbox: tuple[int, int, int, int],
    expected_shape: tuple[int, int, int],
    expected_slice: tuple[slice, slice],
) -> None:
    """Les crops près des bords restent dans l'image et contiennent les pixels attendus."""
    image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)

    crop = crop_face(image, bbox, margin=0.5)

    assert crop.shape == expected_shape
    np.testing.assert_array_equal(crop, image[expected_slice[0], expected_slice[1]])


def test_crop_face_keeps_a_tiny_box_at_the_border_nonempty() -> None:
    """Une petite boîte près du coin est décalée dans l'image au lieu d'être vide."""
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)

    crop = crop_face(image, (0.01, 0.01, 0.11, 0.11), margin=0)

    assert crop.shape == (1, 1, 3)
    np.testing.assert_array_equal(crop[0, 0], image[0, 0])


@pytest.mark.parametrize(
    ("bbox", "margin"),
    [
        ((1, 1, 1, 5), 0.25),
        ((1, 1, 5, 5), -0.1),
        ((1, 1, 5, 5), float("inf")),
        ((20, 1, 25, 5), 0.25),
        ((1, 1, float("nan"), 5), 0.25),
    ],
)
def test_crop_face_rejects_invalid_box_or_margin(
    bbox: tuple[float, float, float, float],
    margin: float,
) -> None:
    """Les boîtes et marges invalides sont signalées plutôt que masquées."""
    image = np.zeros((10, 12, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        crop_face(image, bbox, margin)
