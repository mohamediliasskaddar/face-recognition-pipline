from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Erreur de structure ou de valeur dans le fichier de configuration."""


@dataclass(frozen=True)
class ModelConfig:
    """Paramètres du modèle de détection et de reconnaissance."""

    backend: str
    name: str
    det_size: tuple[int, int]
    providers: tuple[str, ...]


@dataclass(frozen=True)
class DetectionConfig:
    """Seuils de filtrage des visages détectés."""

    min_det_score: float
    min_face_size: int


@dataclass(frozen=True)
class MatchingConfig:
    """Paramètres de comparaison et de regroupement des identités."""

    threshold: float
    use_query_expansion: bool
    expansion_top_k: int
    use_gallery_clustering: bool
    cluster_distance_threshold: float


@dataclass(frozen=True)
class QualityConfig:
    """Seuils de fiabilité des visages."""

    min_det_score: float = 0.6
    min_face_size: int = 50
    min_sharpness: float = 40.0


@dataclass(frozen=True)
class EmbeddingConfig:
    """Options de prétraitement des visages avant l'embedding."""

    use_flip: bool = True
    use_clahe: bool = False


@dataclass(frozen=True)
class ClusteringConfig:
    """Seuils de clustering et de rattachement des visages."""

    distance_threshold: float = 0.6
    attach_threshold: float = 0.35
    merge_threshold: float = 0.55


@dataclass(frozen=True)
class OutputConfig:
    """Options d'export des visages regroupés."""

    crop_margin: float = 0.25
    copy_originals: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class PathsConfig:
    """Chemins de sortie et de cache du pipeline."""

    cache_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class AppConfig:
    """Configuration complète de l'application."""

    model: ModelConfig
    detection: DetectionConfig
    matching: MatchingConfig
    paths: PathsConfig
    quality: QualityConfig = field(default_factory=QualityConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"La section '{name}' doit être un dictionnaire.")
    return value


def _value(section: dict[str, Any], key: str, expected_type: type) -> Any:
    value = section.get(key)
    if expected_type is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, expected_type) or (expected_type is int and isinstance(value, bool)):
        raise ConfigError(f"La valeur '{key}' doit être de type {expected_type.__name__}.")
    return value


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Charge le YAML et retourne sa représentation en dataclasses typées.

    Les chemins relatifs définis dans le YAML sont résolus depuis la racine du
    projet. En l'absence de chemin explicite, le fichier `configs/config.yaml`
    du projet est utilisé.

    Args:
        config_path: Chemin du fichier YAML. Par défaut, la configuration du projet.

    Returns:
        Configuration validée de l'application.

    Raises:
        FileNotFoundError: Si le fichier de configuration n'existe pas.
        ConfigError: Si le YAML ou une de ses valeurs est invalide.
    """
    project_root = Path(__file__).resolve().parents[1]
    source_path = (
        Path(config_path) if config_path is not None else (project_root / "configs" / "config.yaml")
    )

    try:
        with source_path.open(encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ConfigError(f"Le fichier YAML '{source_path}' est invalide: {error}") from error

    if not isinstance(data, dict):
        raise ConfigError("La configuration doit contenir un dictionnaire YAML.")

    model_data = _section(data, "model")
    detection_data = _section(data, "detection")
    matching_data = _section(data, "matching")
    paths_data = _section(data, "paths")
    quality_data = _section(data, "quality")
    embedding_data = _section(data, "embedding")
    clustering_data = _section(data, "clustering")
    output_data = _section(data, "output")

    det_size = model_data.get("det_size")
    if (
        not isinstance(det_size, list)
        or len(det_size) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) for size in det_size)
    ):
        raise ConfigError("'det_size' doit contenir deux entiers.")

    providers = model_data.get("providers")
    if not isinstance(providers, list) or any(
        not isinstance(provider, str) for provider in providers
    ):
        raise ConfigError("'providers' doit être une liste de chaînes de caractères.")

    min_det_score = _value(quality_data, "min_det_score", float)
    min_face_size = _value(quality_data, "min_face_size", int)
    min_sharpness = _value(quality_data, "min_sharpness", float)
    if not 0.0 < min_det_score <= 1.0:
        raise ConfigError(
            "'quality.min_det_score' doit être supérieur à 0 et inférieur ou égal à 1."
        )
    if min_face_size <= 0:
        raise ConfigError("'quality.min_face_size' doit être un entier positif.")
    if min_sharpness <= 0.0:
        raise ConfigError("'quality.min_sharpness' doit être strictement positif.")

    distance_threshold = _value(clustering_data, "distance_threshold", float)
    attach_threshold = _value(clustering_data, "attach_threshold", float)
    merge_threshold = _value(clustering_data, "merge_threshold", float)
    if not 0.0 <= distance_threshold <= 2.0:
        raise ConfigError("'clustering.distance_threshold' doit être compris entre 0 et 2.")
    if not -1.0 <= attach_threshold <= 1.0:
        raise ConfigError("'clustering.attach_threshold' doit être compris entre -1 et 1.")
    if not -1.0 <= merge_threshold <= 1.0:
        raise ConfigError("'clustering.merge_threshold' doit être compris entre -1 et 1.")

    crop_margin = _value(output_data, "crop_margin", float)
    if not math.isfinite(crop_margin) or crop_margin < 0.0:
        raise ConfigError("'output.crop_margin' doit être positif ou nul.")

    def project_path(key: str) -> Path:
        configured_path = Path(_value(paths_data, key, str))
        if configured_path.is_absolute():
            return configured_path
        return project_root / configured_path

    return AppConfig(
        model=ModelConfig(
            backend=_value(model_data, "backend", str),
            name=_value(model_data, "name", str),
            det_size=(det_size[0], det_size[1]),
            providers=tuple(providers),
        ),
        detection=DetectionConfig(
            min_det_score=_value(detection_data, "min_det_score", float),
            min_face_size=_value(detection_data, "min_face_size", int),
        ),
        matching=MatchingConfig(
            threshold=_value(matching_data, "threshold", float),
            use_query_expansion=_value(matching_data, "use_query_expansion", bool),
            expansion_top_k=_value(matching_data, "expansion_top_k", int),
            use_gallery_clustering=_value(matching_data, "use_gallery_clustering", bool),
            cluster_distance_threshold=_value(matching_data, "cluster_distance_threshold", float),
        ),
        paths=PathsConfig(
            cache_dir=project_path("cache_dir"),
            output_dir=project_path("output_dir"),
        ),
        quality=QualityConfig(
            min_det_score=min_det_score,
            min_face_size=min_face_size,
            min_sharpness=min_sharpness,
        ),
        embedding=EmbeddingConfig(
            use_flip=_value(embedding_data, "use_flip", bool),
            use_clahe=_value(embedding_data, "use_clahe", bool),
        ),
        clustering=ClusteringConfig(
            distance_threshold=distance_threshold,
            attach_threshold=attach_threshold,
            merge_threshold=merge_threshold,
        ),
        output=OutputConfig(
            crop_margin=crop_margin,
            copy_originals=_value(output_data, "copy_originals", bool),
            overwrite=_value(output_data, "overwrite", bool),
        ),
    )
