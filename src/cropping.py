"""Recadrage des visages à partir de leurs boîtes de détection."""

import math

import numpy as np
from numpy.typing import NDArray


def crop_face(
    img: NDArray[np.uint8],
    bbox: tuple[float, float, float, float],
    margin: float = 0.25,
) -> NDArray[np.uint8]:
    """Recadre un visage BGR avec une marge, en restant dans l'image.

    Le recadrage est carré avant clipping aux limites de l'image. La marge est
    ajoutée de chaque côté, en proportion de la plus grande dimension de la boîte.

    Args:
        img: Image BGR uint8 de forme `(hauteur, largeur, 3)`.
        bbox: Coordonnées `(x1, y1, x2, y2)` du visage.
        margin: Marge ajoutée de chaque côté, exprimée en proportion.

    Returns:
        Copie BGR du visage recadré.

    Raises:
        ValueError: Si l'image, la boîte ou la marge est invalide, ou si la boîte
            ne recoupe pas l'image.
    """
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("img doit être une image BGR uint8 de forme (hauteur, largeur, 3).")
    image_height, image_width = img.shape[:2]
    if image_height == 0 or image_width == 0:
        raise ValueError("img ne peut pas être vide.")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin doit être un nombre fini positif ou nul.")

    try:
        coordinates = np.asarray(bbox, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("bbox doit contenir quatre coordonnées numériques finies.") from error
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        raise ValueError("bbox doit contenir quatre coordonnées numériques finies.")

    x1, y1, x2, y2 = (float(coordinate) for coordinate in coordinates)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise ValueError("bbox doit avoir une largeur et une hauteur positives.")
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        raise ValueError("bbox ne recoupe pas l'image.")

    side = math.ceil(max(box_width, box_height) * (1.0 + 2.0 * margin))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    left = min(max(0, left), max(0, image_width - side))
    top = min(max(0, top), max(0, image_height - side))
    right = min(image_width, left + side)
    bottom = min(image_height, top + side)
    return img[top:bottom, left:right].copy()
