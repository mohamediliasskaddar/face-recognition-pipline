"""Tests de lecture, d'écriture et de liste des images."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.io_utils import ImageReadError, ImageWriteError, list_images, read_image, save_image


def test_list_images_filters_extensions_and_sorts_names(tmp_path: Path) -> None:
    """Le listing accepte les extensions voulues et trie sans tenir compte de la casse."""
    for filename in ("b.jpeg", "A.JPG", "d.webp", "c.bmp", "e.png", "ignored.gif"):
        (tmp_path / filename).touch()
    (tmp_path / "folder.png").mkdir()

    assert [path.name for path in list_images(tmp_path)] == [
        "A.JPG",
        "b.jpeg",
        "c.bmp",
        "d.webp",
        "e.png",
    ]


def test_list_images_rejects_missing_folder(tmp_path: Path) -> None:
    """Un dossier manquant produit une erreur explicite."""
    with pytest.raises(FileNotFoundError, match="n'existe pas"):
        list_images(tmp_path / "missing")


def test_list_images_rejects_file_path(tmp_path: Path) -> None:
    """Un chemin de fichier n'est pas accepté comme dossier d'images."""
    file_path = tmp_path / "image.jpg"
    file_path.touch()

    with pytest.raises(NotADirectoryError, match="n'est pas un dossier"):
        list_images(file_path)


def test_read_image_returns_bgr(tmp_path: Path) -> None:
    """La lecture convertit les pixels RGB de Pillow en BGR."""
    image_path = tmp_path / "red.png"
    Image.new("RGB", (2, 3), color=(255, 0, 0)).save(image_path)

    image = read_image(image_path)

    assert image.shape == (3, 2, 3)
    np.testing.assert_array_equal(image[0, 0], np.array([0, 0, 255], dtype=np.uint8))


def test_read_image_corrects_exif_orientation(tmp_path: Path) -> None:
    """L'orientation EXIF 6 pivote l'image de 90 degrés dans le sens horaire."""
    image_path = tmp_path / "rotated.jpg"
    source = Image.new("RGB", (2, 3), color=(20, 80, 160))
    exif = source.getexif()
    exif[274] = 6
    source.save(image_path, exif=exif)

    assert read_image(image_path).shape == (2, 3, 3)


def test_read_image_raises_clear_error_for_corrupt_file(tmp_path: Path) -> None:
    """Un fichier non décodable produit une erreur qui nomme son chemin."""
    image_path = tmp_path / "corrupt.jpg"
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ImageReadError, match="Impossible de lire l'image"):
        read_image(image_path)


def test_save_image_creates_parent_and_writes_bgr(tmp_path: Path) -> None:
    """L'enregistrement crée le dossier parent et conserve les canaux BGR."""
    image_path = tmp_path / "nested" / "blue.png"
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)

    save_image(image_path, image)

    assert image_path.is_file()
    np.testing.assert_array_equal(read_image(image_path), image)


def test_save_image_raises_for_unsupported_extension(tmp_path: Path) -> None:
    """Une extension non prise en charge ne produit pas un faux succès."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ImageWriteError, match="Impossible d'enregistrer l'image"):
        save_image(tmp_path / "image.unsupported", image)
