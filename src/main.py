"""Interface CLI pour lancer une recherche dans la galerie."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from loguru import logger

from src.config import AppConfig, load_config
from src.pipeline import run, run_cluster


def _build_parser() -> argparse.ArgumentParser:
    """Construit le parseur des options CLI."""
    parser = argparse.ArgumentParser(
        description="Trie les visages par personne ou recherche une identité dans une galerie."
    )
    parser.add_argument(
        "--mode",
        choices=("cluster", "search"),
        default="search",
        help="Mode cluster (défaut) ou recherche historique.",
    )
    parser.add_argument("--input", type=Path, help="Dossier source pour le mode cluster.")
    parser.add_argument("--query", type=Path, help="Image de requête pour le mode search.")
    parser.add_argument(
        "--gallery",
        type=Path,
        help="Dossier de galerie pour le mode search.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Dossier de sortie pour matches/ et report.json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Seuil du mode choisi : distance cosinus [0, 2] pour cluster, "
            "similarité cosinus [-1, 1] pour search."
        ),
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        help="Taille minimale du visage en pixels.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        help="Marge relative ajoutée aux crops du mode cluster.",
    )
    copy_group = parser.add_mutually_exclusive_group()
    copy_group.add_argument(
        "--copy-originals",
        dest="copy_originals",
        action="store_true",
        help="Copie l'image source dans le dossier de chaque personne.",
    )
    copy_group.add_argument(
        "--no-copy-originals",
        dest="copy_originals",
        action="store_false",
        help="Ne copie pas les images sources.",
    )
    parser.set_defaults(copy_originals=None)
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Remplace le dossier de sortie du mode cluster.",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Refuse de remplacer un dossier de sortie non vide.",
    )
    parser.set_defaults(overwrite=None)
    parser.add_argument(
        "--face-index",
        type=int,
        help="Index du visage de la requête à rechercher (indexation à partir de 0).",
    )
    parser.add_argument(
        "--backend",
        help="Backend modèle à utiliser (seul 'insightface' est actuellement implémenté).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Reconstruit l'index galerie sans lire ni écrire le cache.",
    )
    return parser


def _override_config(
    config: AppConfig,
    *,
    threshold: float | None,
    backend: str | None,
    mode: str = "search",
    min_face_size: int | None = None,
    margin: float | None = None,
    copy_originals: bool | None = None,
    overwrite: bool | None = None,
) -> AppConfig:
    """Applique les options CLI facultatives à la configuration."""
    matching = config.matching
    clustering = config.clustering
    model = config.model
    detection = config.detection
    quality = config.quality
    output = config.output
    if threshold is not None:
        if mode == "cluster":
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 2.0:
                raise ValueError("--threshold en mode cluster doit être compris entre 0 et 2.")
            clustering = replace(clustering, distance_threshold=threshold)
        else:
            if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
                raise ValueError("--threshold en mode search doit être compris entre -1 et 1.")
            matching = replace(matching, threshold=threshold)
    if backend is not None:
        model = replace(model, backend=backend)
    if min_face_size is not None:
        if min_face_size <= 0:
            raise ValueError("--min-face-size doit être strictement positif.")
        detection = replace(detection, min_face_size=min_face_size)
        quality = replace(quality, min_face_size=min_face_size)
    if margin is not None:
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("--margin doit être un nombre fini positif ou nul.")
        output = replace(output, crop_margin=margin)
    if copy_originals is not None:
        output = replace(output, copy_originals=copy_originals)
    if overwrite is not None:
        output = replace(output, overwrite=overwrite)
    return replace(
        config,
        matching=matching,
        model=model,
        detection=detection,
        quality=quality,
        clustering=clustering,
        output=output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Exécute le mode demandé et retourne un code non nul en cas d'échec."""
    args = _build_parser().parse_args(argv)
    try:
        config = _override_config(
            load_config(),
            threshold=args.threshold,
            backend=args.backend,
            mode=args.mode,
            min_face_size=args.min_face_size,
            margin=args.margin,
            copy_originals=args.copy_originals,
            overwrite=args.overwrite,
        )
        if args.mode == "cluster":
            if args.input is None:
                raise ValueError("--input est obligatoire avec --mode cluster.")
            if args.query is not None or args.gallery is not None or args.face_index is not None:
                raise ValueError("--query, --gallery et --face-index sont réservés au mode search.")
            if args.threshold is not None:
                logger.info(
                    "Seuil de distance cosinus du clustering : {}.",
                    config.clustering.distance_threshold,
                )
            labels = run_cluster(
                args.input,
                args.output,
                config,
                use_cache=not args.no_cache,
            )
            person_count = len({int(label) for label in labels if label >= 0})
            logger.success(
                "Clustering terminé : {} personne(s). Rapport : '{}'.",
                person_count,
                args.output / "report.json",
            )
            return 0

        if args.input is not None:
            raise ValueError("--input est réservé au mode cluster.")
        if args.query is None or args.gallery is None:
            raise ValueError("--query et --gallery sont obligatoires avec --mode search.")
        if args.margin is not None or args.copy_originals is not None or args.overwrite is not None:
            raise ValueError(
                "--margin, --copy-originals et --overwrite sont réservés au mode cluster."
            )

        matches = run(
            args.query,
            args.gallery,
            args.output,
            config,
            face_index=args.face_index,
            use_cache=not args.no_cache,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        logger.error("Échec de l'exécution : {}", error)
        return 1

    if matches:
        logger.success(
            "Recherche terminée : {} correspondance(s). Rapport : '{}'.",
            len(matches),
            args.output / "report.json",
        )
    else:
        logger.info(
            "Recherche terminée sans correspondance. Rapport : '{}'.",
            args.output / "report.json",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
