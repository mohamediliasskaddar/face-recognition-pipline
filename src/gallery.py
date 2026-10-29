"""Indexation de tous les visages de la galerie et cache local des embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm

from src.detection import FaceDetector
from src.embedding import EMBEDDING_DIMENSION, FaceEmbedder
from src.io_utils import ImageReadError, list_images, read_image
from src.types import FaceRecord

CACHE_FORMAT_VERSION = 1
ALIGNED_FACE_SHAPE = (112, 112, 3)
LANDMARK_SHAPE = (5, 2)


class GalleryCacheError(ValueError):
    """Erreur levée si le cache de galerie est corrompu ou invalide."""


def _hash_files(folder: Path, image_paths: list[Path]) -> str:
    """Calcule l'empreinte SHA-256 des noms et contenus des images."""
    digest = hashlib.sha256()
    for image_path in image_paths:
        relative_path = image_path.relative_to(folder).as_posix().encode("utf-8")
        stat = image_path.stat()
        digest.update(len(relative_path).to_bytes(8, byteorder="big"))
        digest.update(relative_path)
        digest.update(stat.st_mtime_ns.to_bytes(8, byteorder="big", signed=True))
        digest.update(stat.st_size.to_bytes(8, byteorder="big"))
        with image_path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _configuration_fingerprint(detector: FaceDetector, embedder: FaceEmbedder) -> str:
    """Empreinte les paramètres qui influencent les faces et embeddings calculés."""
    detector_config = detector.config
    embedder_config = embedder.config
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "detector": {
            "backend": detector_config.model.backend,
            "name": detector_config.model.name,
            "det_size": detector_config.model.det_size,
            "providers": detector_config.model.providers,
            "min_det_score": detector_config.detection.min_det_score,
            "min_face_size": detector_config.detection.min_face_size,
        },
        "embedder": {
            "backend": embedder_config.model.backend,
            "name": embedder_config.model.name,
            "embedder_class": (f"{type(embedder).__module__}.{type(embedder).__qualname__}"),
        },
    }
    encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def _validate_records(records: list[FaceRecord]) -> None:
    """Vérifie la structure de chaque enregistrement avant sauvegarde."""
    for index, record in enumerate(records):
        if len(record.bbox) != 4:
            raise GalleryCacheError(f"La boîte du visage {index} doit contenir 4 coordonnées.")
        if record.landmarks.shape != LANDMARK_SHAPE:
            raise GalleryCacheError(
                f"Les landmarks du visage {index} doivent avoir la forme {LANDMARK_SHAPE}."
            )
        if record.aligned.shape != ALIGNED_FACE_SHAPE or record.aligned.dtype != np.uint8:
            raise GalleryCacheError(
                f"Le crop du visage {index} doit être uint8 de forme {ALIGNED_FACE_SHAPE}."
            )
        if record.embedding.shape != (EMBEDDING_DIMENSION,):
            raise GalleryCacheError(
                f"L'embedding du visage {index} doit avoir la forme ({EMBEDDING_DIMENSION},)."
            )
        if record.landmarks.dtype != np.float32 or record.embedding.dtype != np.float32:
            raise GalleryCacheError(
                f"Les landmarks et l'embedding du visage {index} doivent être float32."
            )
        if not np.isfinite(record.landmarks).all() or not np.isfinite(record.embedding).all():
            raise GalleryCacheError(
                f"Les landmarks et l'embedding du visage {index} doivent être finis."
            )
        embedding_norm = float(np.linalg.norm(record.embedding.astype(np.float64)))
        if not np.isclose(embedding_norm, 1.0, rtol=1e-5, atol=1e-5):
            raise GalleryCacheError(
                f"L'embedding du visage {index} doit être normalisé L2 (norme attendue : 1)."
            )
        if not np.isfinite(record.det_score):
            raise GalleryCacheError(f"Le score du visage {index} doit être fini.")


def save_cache(
    cache_path: str | Path,
    records: list[FaceRecord],
    *,
    gallery_fingerprint: str,
    configuration_fingerprint: str,
) -> None:
    """Enregistre atomiquement les embeddings et métadonnées dans un NPZ compressé.

    Les données sont stockées sans pickle. Les fichiers temporaires sont créés
    dans le même dossier que le cache pour permettre un remplacement atomique.

    Args:
        cache_path: Destination du cache, généralement `data/cache/embeddings.npz`.
        records: Visages indexés de la galerie.
        gallery_fingerprint: SHA-256 des chemins et contenus de la galerie.
        configuration_fingerprint: SHA-256 des paramètres de détection et d'embedding.

    Raises:
        GalleryCacheError: Si un enregistrement ne respecte pas le format du cache.
        OSError: Si le cache ne peut pas être écrit.
    """
    _validate_records(records)
    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = len(records)

    metadata = json.dumps(
        {
            "format_version": CACHE_FORMAT_VERSION,
            "gallery_fingerprint": gallery_fingerprint,
            "configuration_fingerprint": configuration_fingerprint,
        },
        sort_keys=True,
    )
    arrays: dict[str, NDArray[Any]] = {
        "metadata": np.asarray(metadata),
        "image_paths": np.asarray([record.image_path for record in records], dtype=np.str_),
        "bboxes": np.asarray([record.bbox for record in records], dtype=np.int64).reshape(count, 4),
        "landmarks": np.asarray([record.landmarks for record in records], dtype=np.float32).reshape(
            count, *LANDMARK_SHAPE
        ),
        "det_scores": np.asarray([record.det_score for record in records], dtype=np.float32),
        "aligned_faces": np.asarray([record.aligned for record in records], dtype=np.uint8).reshape(
            count, *ALIGNED_FACE_SHAPE
        ),
        "embeddings": np.asarray(
            [record.embedding for record in records], dtype=np.float32
        ).reshape(count, EMBEDDING_DIMENSION),
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_file, **arrays)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_cache(
    cache_path: str | Path,
    *,
    gallery_fingerprint: str,
    configuration_fingerprint: str,
) -> list[FaceRecord] | None:
    """Charge le cache s'il correspond à la galerie et à la configuration actuelles.

    Un cache manquant ou périmé retourne `None` afin que l'indexation soit recalculée.
    Un cache présent mais illisible ou mal formé lève `GalleryCacheError`.

    Args:
        cache_path: Fichier NPZ de cache.
        gallery_fingerprint: SHA-256 attendu pour les chemins et contenus de la galerie.
        configuration_fingerprint: SHA-256 attendu pour les paramètres de modèle.

    Returns:
        Enregistrements restaurés, ou `None` si le cache manque ou est périmé.

    Raises:
        GalleryCacheError: Si le cache existant est corrompu ou mal formé.
    """
    source = Path(cache_path)
    if not source.is_file():
        return None

    try:
        with np.load(source, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"].item()))
            if (
                metadata.get("format_version") != CACHE_FORMAT_VERSION
                or metadata.get("gallery_fingerprint") != gallery_fingerprint
                or metadata.get("configuration_fingerprint") != configuration_fingerprint
            ):
                return None

            image_paths = cache["image_paths"]
            bboxes = cache["bboxes"]
            landmarks = cache["landmarks"]
            det_scores = cache["det_scores"]
            aligned_faces = cache["aligned_faces"]
            embeddings = cache["embeddings"]
            count = len(image_paths)

            if (
                image_paths.ndim != 1
                or bboxes.shape != (count, 4)
                or landmarks.shape != (count, *LANDMARK_SHAPE)
                or det_scores.shape != (count,)
                or aligned_faces.shape != (count, *ALIGNED_FACE_SHAPE)
                or embeddings.shape != (count, EMBEDDING_DIMENSION)
                or image_paths.dtype.kind not in {"U", "S"}
                or bboxes.dtype.kind not in {"i", "u"}
                or landmarks.dtype != np.float32
                or det_scores.dtype.kind != "f"
                or aligned_faces.dtype != np.uint8
                or embeddings.dtype != np.float32
            ):
                raise GalleryCacheError(
                    f"Les tableaux du cache '{source}' ont des formes invalides."
                )

            records = [
                FaceRecord(
                    image_path=str(image_paths[index]),
                    bbox=tuple(int(value) for value in bboxes[index]),
                    landmarks=np.asarray(landmarks[index], dtype=np.float32),
                    det_score=float(det_scores[index]),
                    aligned=np.asarray(aligned_faces[index], dtype=np.uint8),
                    embedding=np.asarray(embeddings[index], dtype=np.float32),
                )
                for index in range(count)
            ]
    except GalleryCacheError:
        raise
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        raise GalleryCacheError(f"Impossible de lire le cache '{source}': {error}") from error

    _validate_records(records)
    return records


def build_gallery_index(
    folder: str | Path,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    *,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
) -> list[FaceRecord]:
    """Indexe tous les visages de la galerie ou réutilise un cache valide.

    Chaque visage détecté est aligné et encodé séparément, de sorte qu'une image
    contenant plusieurs personnes crée plusieurs `FaceRecord`. Les images
    illisibles sont journalisées puis ignorées; les erreurs de détection,
    d'alignement ou d'embedding sont propagées.

    Args:
        folder: Dossier de galerie.
        detector: Détecteur préparé et réutilisable.
        embedder: Extracteur d'embeddings préparé et réutilisable.
        cache_path: Chemin de cache facultatif. Par défaut,
            `detector.config.paths.cache_dir/embeddings.npz`.
        use_cache: Si `False`, reconstruit l'index sans lire ni écrire le cache.

    Returns:
        Un enregistrement par visage valide trouvé dans la galerie.

    Raises:
        FileNotFoundError: Si le dossier de galerie n'existe pas.
        NotADirectoryError: Si le chemin de galerie n'est pas un dossier.
        OSError: Si la lecture du contenu d'un fichier échoue pour une autre raison
            qu'une image illisible.
        ValueError: Si le modèle retourne des données de visage invalides.
    """
    gallery_dir = Path(folder)
    image_paths = list_images(gallery_dir)
    destination: Path | None = None
    gallery_fingerprint: str | None = None
    configuration_fingerprint: str | None = None
    if use_cache:
        gallery_fingerprint = _hash_files(gallery_dir, image_paths)
        configuration_fingerprint = _configuration_fingerprint(detector, embedder)
        destination = (
            Path(cache_path)
            if cache_path is not None
            else detector.config.paths.cache_dir / "embeddings.npz"
        )
        cached_records = load_cache(
            destination,
            gallery_fingerprint=gallery_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
        )
        if cached_records is not None:
            logger.info(
                "Cache de galerie rechargé : {} visages depuis '{}'.",
                len(cached_records),
                destination,
            )
            return cached_records

    records: list[FaceRecord] = []
    for image_path in tqdm(image_paths, desc="Indexation de la galerie", unit="image"):
        try:
            image = read_image(image_path)
        except ImageReadError as error:
            logger.warning("Image de galerie ignorée '{}': {}", image_path, error)
            continue

        faces = detector.detect(image)
        if not faces:
            logger.warning("Aucun visage détecté dans '{}'.", image_path)
            continue

        for face_index, face in enumerate(faces):
            landmarks = np.asarray(face.kps, dtype=np.float32)
            aligned = detector.align(image, landmarks)
            embedding = np.asarray(
                embedder.embed_detected_face(face, aligned),
                dtype=np.float32,
            )
            bbox_coordinates = [round(float(coordinate)) for coordinate in face.bbox]
            bbox = (
                bbox_coordinates[0],
                bbox_coordinates[1],
                bbox_coordinates[2],
                bbox_coordinates[3],
            )
            records.append(
                FaceRecord(
                    image_path=str(image_path),
                    bbox=bbox,
                    landmarks=landmarks,
                    det_score=float(face.det_score),
                    aligned=aligned,
                    embedding=embedding,
                )
            )
            logger.debug(
                "Visage {} indexé depuis '{}' (score={:.3f}).",
                face_index + 1,
                image_path,
                face.det_score,
            )

    if use_cache:
        if destination is None or gallery_fingerprint is None or configuration_fingerprint is None:
            raise RuntimeError("Les métadonnées de cache ne sont pas initialisées.")
        save_cache(
            destination,
            records,
            gallery_fingerprint=gallery_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
        )
        logger.info(
            "Index de galerie créé : {} visages issus de {} images; cache écrit dans '{}'.",
            len(records),
            len(image_paths),
            destination,
        )
    else:
        logger.info(
            "Index de galerie créé sans cache : {} visages issus de {} images.",
            len(records),
            len(image_paths),
        )
    return records
