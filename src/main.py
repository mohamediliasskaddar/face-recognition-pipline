"""Interface CLI pour lancer une recherche dans la galerie."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from loguru import logger

from src.config import AppConfig, load_config
from src.pipeline import run


def _build_parser() -> argparse.ArgumentParser:
    """Construit le parseur des options CLI."""
    parser = argparse.ArgumentParser(
        description="Recherche les images de galerie correspondant à une personne."
    )
    parser.add_argument("--query", required=True, type=Path, help="Image de requête.")
    parser.add_argument(
        "--gallery",
        required=True,
        type=Path,
        help="Dossier contenant les images de galerie.",
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
        help="Seuil cosinus, entre -1 et 1 (par défaut : config.yaml).",
    )
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
) -> AppConfig:
    """Applique les options CLI facultatives à la configuration."""
    matching = config.matching
    model = config.model
    if threshold is not None:
        if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("--threshold doit être un nombre fini compris entre -1 et 1.")
        matching = replace(matching, threshold=threshold)
    if backend is not None:
        model = replace(model, backend=backend)
    return replace(config, matching=matching, model=model)


def main(argv: Sequence[str] | None = None) -> int:
    """Exécute le CLI et retourne un code non nul si la recherche échoue."""
    args = _build_parser().parse_args(argv)
    try:
        config = _override_config(
            load_config(),
            threshold=args.threshold,
            backend=args.backend,
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
        logger.error("Échec de la recherche : {}", error)
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
