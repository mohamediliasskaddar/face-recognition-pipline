"""Utilitaires de lecture et d'écriture des images."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageReadError(OSError):
    """Erreur levée lorsqu'une image ne peut pas être lue ou décodée."""


class ImageWriteError(OSError):
    """Erreur levée lorsqu'une image ne peut pas être enregistrée."""


def list_images(folder: str | Path) -> list[Path]:
    """Liste les images d'un dossier, triées alphabétiquement par nom.

    Les extensions acceptées sont jpg, jpeg, png, bmp et webp, sans distinction
    de casse. Seuls les fichiers directement présents dans le dossier sont pris
    en compte.

    Args:
        folder: Dossier à parcourir.

    Returns:
        Chemins des images trouvées, dans l'ordre alphabétique.

    Raises:
        FileNotFoundError: Si le dossier n'existe pas.
        NotADirectoryError: Si le chemin ne désigne pas un dossier.
    """
    directory = Path(folder)
    if not directory.exists():
        raise FileNotFoundError(f"Le dossier d'images n'existe pas : '{directory}'.")
    if not directory.is_dir():
        raise NotADirectoryError(f"Le chemin n'est pas un dossier : '{directory}'.")

    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: (path.name.casefold(), path.name))


def read_image(path: str | Path) -> NDArray[np.uint8]:
    """Lit une image, corrige son orientation EXIF et la retourne en BGR.

    Args:
        path: Chemin de l'image à lire.

    Returns:
        Image décodée au format BGR sous forme de tableau uint8.

    Raises:
        ImageReadError: Si le fichier est illisible ou n'est pas une image décodable.
    """
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            oriented_image = ImageOps.exif_transpose(source).convert("RGB")
            image_rgb = np.asarray(oriented_image, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ImageReadError(f"Impossible de lire l'image '{image_path}': {error}") from error

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def save_image(path: str | Path, image: NDArray[np.uint8]) -> None:
    """Enregistre une image BGR et crée son dossier parent si nécessaire.

    Args:
        path: Chemin de destination, dont l'extension doit être prise en charge par OpenCV.
        image: Image BGR à enregistrer.

    Raises:
        ImageWriteError: Si OpenCV ne parvient pas à enregistrer l'image.
        OSError: Si le dossier parent ne peut pas être créé.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(destination), image)
    except cv2.error as error:
        raise ImageWriteError(
            f"Impossible d'enregistrer l'image dans '{destination}': {error}"
        ) from error
    if not written:
        raise ImageWriteError(f"Impossible d'enregistrer l'image dans '{destination}'.")
