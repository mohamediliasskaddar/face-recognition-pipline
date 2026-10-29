"""Tests des artefacts ZIP produits par l'interface Streamlit."""

import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from app import streamlit_app
from src.config import AppConfig, load_config
from src.types import MatchResult


def test_build_results_archive_includes_matches_annotations_and_reports(
    tmp_path: Path,
) -> None:
    """L'archive conserve les chemins relatifs et inclut les artefacts attendus."""
    (tmp_path / "matches").mkdir()
    (tmp_path / "annotated").mkdir()
    files = {
        "matches/alice.jpg": b"matched image",
        "annotated/alice.jpg": b"annotated image",
        "annotated/query_upload.png": b"annotated query",
        "result_grid.png": b"grid",
        "similarity_heatmap.png": b"heatmap",
        "report.json": b'{"n_matches": 1}',
    }
    for relative_path, content in files.items():
        file_path = tmp_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    archive_bytes = streamlit_app.build_results_archive(tmp_path)

    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == set(files)
        assert {name: archive.read(name) for name in archive.namelist()} == files


def test_build_results_archive_rejects_missing_pipeline_output(tmp_path: Path) -> None:
    """Une archive vide n'est jamais présentée comme un téléchargement valide."""
    with pytest.raises(FileNotFoundError, match="Aucun résultat exportable"):
        streamlit_app.build_results_archive(tmp_path)


def test_image_validation_accepts_real_fixture_and_rejects_corrupt_bytes() -> None:
    """L'aperçu accepte une image décodable et refuse les données non-image."""
    fixture_path = Path(__file__).parent / "fixtures" / "pic-with-no-face.png"

    assert streamlit_app._is_valid_image(fixture_path.read_bytes())
    assert not streamlit_app._is_valid_image(b"this is not an image")
    assert not streamlit_app._is_valid_image(b"")


def test_run_search_calls_pipeline_and_packages_its_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le helper transmet seuil/cache au pipeline et conserve ses artefacts."""
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    matched_image = gallery / "match.jpg"
    assert cv2.imwrite(str(matched_image), np.zeros((8, 8, 3), dtype=np.uint8))
    pipeline_call: dict[str, object] = {}

    def fake_run(
        query_path: Path,
        gallery_dir: Path,
        output_dir: Path,
        config: AppConfig,
        *,
        use_cache: bool,
    ) -> list[MatchResult]:
        pipeline_call.update(
            query=query_path,
            gallery=gallery_dir,
            output=output_dir,
            threshold=config.matching.threshold,
            use_cache=use_cache,
            query_bytes=query_path.read_bytes(),
            query_suffix=query_path.suffix,
        )
        (output_dir / "matches").mkdir(parents=True)
        (output_dir / "annotated").mkdir()
        (output_dir / "matches" / matched_image.name).write_bytes(b"match")
        (output_dir / "annotated" / "query.png").write_bytes(b"annotated")
        (output_dir / "result_grid.png").write_bytes(b"grid")
        (output_dir / "report.json").write_text('{"n_matches": 1}', encoding="utf-8")
        return [MatchResult(str(matched_image), 0.9, 0)]

    monkeypatch.setattr(streamlit_app, "run", fake_run)
    config = load_config()
    uploaded_bytes = b"uploaded image bytes"

    matches, grid_bytes, archive_bytes = streamlit_app._run_search(
        uploaded_bytes,
        "query.png",
        gallery,
        config,
        0.73,
        False,
    )

    assert pipeline_call["gallery"] == gallery
    assert pipeline_call["threshold"] == 0.73
    assert pipeline_call["use_cache"] is False
    assert pipeline_call["query_bytes"] == uploaded_bytes
    assert pipeline_call["query_suffix"] == ".png"
    assert matches == [
        {
            "filename": "match.jpg",
            "image_path": str(matched_image),
            "score": 0.9,
            "matched_face_index": 0,
        }
    ]
    assert grid_bytes == b"grid"
    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert "matches/match.jpg" in archive.namelist()
        assert "result_grid.png" in archive.namelist()


def test_streamlit_page_renders_without_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """La page se lance et expose les contrôles principaux dans Streamlit."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert [element.value for element in app.title] == ["Recherche faciale dans une galerie locale"]
    assert [element.label for element in app.slider] == ["Seuil de similarité cosinus"]
    assert len(app.get("file_uploader")) == 1


def test_streamlit_script_imports_when_started_outside_project_root(
    tmp_path: Path,
) -> None:
    """Les imports src fonctionnent même si le cwd n'est pas la racine du dépôt."""
    project_root = Path(__file__).parents[1]
    app_path = project_root / "app" / "streamlit_app.py"
    script = (
        "import runpy, sys\n"
        "from pathlib import Path\n"
        "app_path, project_root = map(Path, sys.argv[1:3])\n"
        "root = str(project_root.resolve())\n"
        "sys.path = [entry for entry in sys.path if Path(entry or '.').resolve() != project_root]\n"
        "runpy.run_path(str(app_path), run_name='streamlit_app_import_test')\n"
        "assert root in sys.path\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(app_path), str(project_root)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
