"""Tests du chargement et de la validation de la configuration."""

from pathlib import Path

from src.config import load_config


def test_load_config_uses_typed_values() -> None:
    """La configuration par défaut est chargée avec des types explicites."""
    config = load_config()

    assert config.model.det_size == (640, 640)
    assert config.model.providers == ("CPUExecutionProvider",)
    assert config.detection.min_det_score == 0.5
    assert isinstance(config.paths.cache_dir, Path)
    assert config.quality.min_det_score == 0.6
    assert config.quality.min_face_size == 50
    assert config.quality.min_sharpness == 40.0
    assert config.embedding.use_flip is True
    assert config.embedding.use_clahe is False
    assert config.clustering.distance_threshold == 0.6
    assert config.clustering.attach_threshold == 0.35
    assert config.clustering.merge_threshold == 0.55
    assert config.output.crop_margin == 0.25
    assert config.output.copy_originals is False
    assert config.output.overwrite is True
