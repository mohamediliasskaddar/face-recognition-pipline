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
