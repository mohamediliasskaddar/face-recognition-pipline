"""Orchestration du pipeline de reconnaissance faciale."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from src.config import AppConfig
from src.detection import FaceDetector, select_main_face
from src.embedding import InsightFaceEmbedder
from src.gallery import build_gallery_index
from src.identity import resolve_identity
from src.io_utils import list_images, read_image
from src.types import MatchResult


class PipelineError(RuntimeError):
    """Erreur métier empêchant une recherche de galerie de s'exécuter."""


def _select_query_face_index(face_index: int | None, face_count: int) -> int | None:
    """Valide l'index choisi; ``None`` conserve la sélection automatique."""
    if face_index is None:
        return None
    if face_index < 0 or face_index >= face_count:
        raise PipelineError(
            f"L'index de visage {face_index} est invalide : "
            f"la requête contient {face_count} visage(s), index attendus "
            f"entre 0 et {face_count - 1}."
        )
    return face_index


def _write_report(
    report_path: Path,
    query_path: Path,
    threshold: float,
    gallery_image_count: int,
    matches: list[MatchResult],
) -> None:
    """Écrit le rapport JSON de la recherche."""
    report = {
        "query": str(query_path),
        "threshold": threshold,
        "n_gallery_images": gallery_image_count,
        "n_matches": len(matches),
        "matches": [
            {
                "image": match.image_path,
                "score": match.best_score,
                "matched_face_index": match.matched_face_index,
            }
            for match in matches
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _copy_matches(matches: list[MatchResult], matches_dir: Path) -> None:
    """Copie les images retenues dans le dossier de sortie."""
    matches_dir.mkdir(parents=True, exist_ok=True)
    for match in matches:
        source = Path(match.image_path)
        if not source.is_file():
            raise PipelineError(f"L'image correspondante n'existe plus : '{source}'.")
        shutil.copy2(source, matches_dir / source.name)


def run(
    query_path: str | Path,
    gallery_dir: str | Path,
    output_dir: str | Path,
    cfg: AppConfig,
    *,
    face_index: int | None = None,
    use_cache: bool = True,
) -> list[MatchResult]:
    """Exécute une recherche faciale et exporte les images et le rapport.

    Args:
        query_path: Image de requête.
        gallery_dir: Dossier contenant les images de référence.
        output_dir: Dossier où créer `matches/` et `report.json`.
        cfg: Configuration de détection, modèle et stratégies de recherche.
        face_index: Index de visage de la requête; par défaut, le visage principal.
        use_cache: Si `False`, reconstruit l'index sans cache.

    Returns:
        Les images correspondantes triées par score décroissant. Une liste vide
        signifie qu'aucune image n'a dépassé les stratégies configurées.

    Raises:
        PipelineError: Si aucun visage n'est détecté, l'index est vide ou l'index
            de visage demandé est invalide.
        OSError, ValueError, RuntimeError: Si la lecture, les modèles ou le cache
            signalent une erreur.
    """
    if cfg.model.backend != "insightface":
        raise PipelineError(
            f"Le backend '{cfg.model.backend}' n'est pas implémenté; "
            "le backend disponible est 'insightface'."
        )
    if not -1.0 <= cfg.matching.threshold <= 1.0:
        raise ValueError("Le seuil cosinus doit être compris entre -1 et 1.")

    query = Path(query_path)
    gallery = Path(gallery_dir)
    output = Path(output_dir)
    logger.info("Chargement de l'image requête '{}'.", query)
    query_image = read_image(query)

    embedder = InsightFaceEmbedder(cfg)
    detector = FaceDetector(cfg, analysis=embedder.analysis)
    query_faces = detector.detect(query_image)
    if not query_faces:
        raise PipelineError(f"Aucun visage détecté dans l'image requête '{query}'.")

    selected_index = _select_query_face_index(face_index, len(query_faces))
    query_face = (
        query_faces[selected_index]
        if selected_index is not None
        else select_main_face(query_faces)
    )
    if len(query_faces) > 1:
        logger.info(
            "{} visage(s) détecté(s) dans la requête; visage choisi : {}.",
            len(query_faces),
            selected_index if selected_index is not None else "principal",
        )
    aligned_face = detector.align(query_image, query_face.kps)
    query_embedding: NDArray[np.float32] = embedder.embed_detected_face(
        query_face,
        aligned_face,
    )

    gallery_images = list_images(gallery)
    if not gallery_images:
        raise PipelineError(f"Aucune image trouvée dans la galerie '{gallery}'.")
    records = build_gallery_index(
        gallery,
        detector,
        embedder,
        use_cache=use_cache,
    )
    if not records:
        raise PipelineError(
            f"Aucun visage exploitable n'a été indexé dans la galerie '{gallery}'."
        )

    matches = resolve_identity(query_embedding, records, cfg)
    output_matches_dir = output / "matches"
    _copy_matches(matches, output_matches_dir)
    _write_report(
        output / "report.json",
        query,
        cfg.matching.threshold,
        len({record.image_path for record in records}),
        matches,
    )
    if matches:
        logger.info(
            "{} image(s) correspondante(s) exportée(s) dans '{}'.",
            len(matches),
            output_matches_dir,
        )
    else:
        logger.info(
            "Aucune correspondance au seuil {}. Rapport écrit dans '{}'.",
            cfg.matching.threshold,
            output / "report.json",
        )
    return matches
