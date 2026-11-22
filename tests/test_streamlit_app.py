"""Tests des artefacts ZIP produits par l'interface Streamlit."""

import ast
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from app import streamlit_app
from app.ui import services
from src.config import AppConfig, load_config
from src.types import MatchResult


def test_brand_header_falls_back_to_text_when_banner_is_missing(tmp_path: Path) -> None:
    """L'en-tête reste affiché même si l'image de bannière n'existe pas."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(
        lambda: streamlit_app.render_header(tmp_path / "missing-banner.png")
    ).run()

    assert not app.exception
    assert [item.value for item in app.title] == ["TrombiAI"]
    assert len(app.get("imgs")) == 0


def test_ui_services_do_not_import_streamlit_or_access_session_state() -> None:
    """Les services restent indépendants de Streamlit et l'état reste encapsulé."""
    app_dir = Path(__file__).parents[1] / "app"
    services_tree = ast.parse((app_dir / "ui" / "services.py").read_text(encoding="utf-8"))
    components_source = (app_dir / "ui" / "components.py").read_text(encoding="utf-8")
    imported_modules = {
        alias.name
        for node in ast.walk(services_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any(
        name == "streamlit" or name.startswith("streamlit.") for name in imported_modules
    )
    assert "st.session_state" not in components_source


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

    archive_bytes = services.build_results_archive(tmp_path)

    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == set(files)
        assert {name: archive.read(name) for name in archive.namelist()} == files


def test_build_results_archive_rejects_missing_pipeline_output(tmp_path: Path) -> None:
    """Une archive vide n'est jamais présentée comme un téléchargement valide."""
    with pytest.raises(FileNotFoundError, match="Aucun résultat exportable"):
        services.build_results_archive(tmp_path)


def test_build_cluster_archive_includes_crops_contact_sheets_and_report(
    tmp_path: Path,
) -> None:
    """L'archive de tri contient les crops, les planches et le rapport."""
    files = {
        "person_01/face0.jpg": b"crop",
        "unknown/face1.jpg": b"unknown",
        "contact_sheets/person_01.png": b"sheet",
        "report.json": b'{"n_faces": 2}',
    }
    for relative_path, content in files.items():
        file_path = tmp_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    archive_bytes = services.build_cluster_archive(tmp_path)

    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == set(files)
        assert {name: archive.read(name) for name in archive.namelist()} == files


def test_image_validation_accepts_real_fixture_and_rejects_corrupt_bytes() -> None:
    """L'aperçu accepte une image décodable et refuse les données non-image."""
    fixture_path = Path(__file__).parent / "fixtures" / "pic-with-no-face.png"

    assert services.is_valid_image(fixture_path.read_bytes())
    assert not services.is_valid_image(b"this is not an image")
    assert not services.is_valid_image(b"")


def test_resolve_labels_csv_uses_default_when_no_upload(tmp_path: Path) -> None:
    """Sans upload, l'application conserve le labels.csv du projet."""
    labels_path, source_label = services.resolve_labels_csv(
        None,
        tmp_path,
    )

    assert labels_path == services.DEFAULT_LABELS_PATH
    assert "data/labels.csv" in source_label


def test_resolve_labels_csv_accepts_uploaded_csv(tmp_path: Path) -> None:
    """Un CSV importé est écrit et validé avant utilisation."""

    class UploadedCsv:
        name = "custom_labels.csv"

        @staticmethod
        def getvalue() -> bytes:
            return b"filename,face_index,person_id\nperson.jpg,0,Person 01\n"

    labels_path, source_label = services.resolve_labels_csv(
        UploadedCsv(),
        tmp_path,
    )

    assert labels_path.read_text(encoding="utf-8").startswith("filename,face_index")
    assert source_label == "upload: custom_labels.csv"


def test_write_uploaded_gallery_preserves_nested_paths(tmp_path: Path) -> None:
    """Les sous-dossiers importés sont conservés sans sortir de la galerie temporaire."""
    image_path = Path(__file__).parent / "fixtures" / "pic-with-no-face.png"
    gallery = tmp_path / "gallery"

    count = services.write_uploaded_gallery(
        [("folder/landscape.png", image_path.read_bytes())],
        gallery,
    )

    assert count == 1
    assert (gallery / "folder" / "landscape.png").read_bytes() == image_path.read_bytes()


@pytest.mark.parametrize(
    ("name", "contents", "message"),
    [
        ("../outside.png", b"image", "Chemin de fichier"),
        ("sub/../outside.png", b"image", "Chemin de fichier"),
        ("image.txt", b"image", "Format d'image"),
        ("corrupt.png", b"not an image", "Image vide"),
    ],
)
def test_write_uploaded_gallery_rejects_invalid_files(
    tmp_path: Path,
    name: str,
    contents: bytes,
    message: str,
) -> None:
    """Les chemins dangereux, formats inconnus et octets corrompus sont refusés."""
    with pytest.raises(ValueError, match=message):
        services.write_uploaded_gallery([(name, contents)], tmp_path / "gallery")


def test_write_uploaded_gallery_rejects_duplicate_paths(tmp_path: Path) -> None:
    """Deux fichiers ne peuvent pas écraser le même chemin importé."""
    image_path = Path(__file__).parent / "fixtures" / "pic-with-no-face.png"
    contents = image_path.read_bytes()

    with pytest.raises(ValueError, match="même chemin"):
        services.write_uploaded_gallery(
            [("same.png", contents), ("same.png", contents)],
            tmp_path / "gallery",
        )


def test_run_cluster_processes_uploaded_directory_and_sanitizes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le helper transmet les réglages et retourne les visuels sans chemins temporaires."""
    image_path = Path(__file__).parent / "fixtures" / "pic-with-no-face.png"
    uploaded_bytes = image_path.read_bytes()
    calls: dict[str, object] = {}

    def fake_run_cluster(
        gallery_dir: Path,
        output_dir: Path,
        config: AppConfig,
        *,
        use_cache: bool,
    ) -> np.ndarray:
        calls.update(
            files=[path.relative_to(gallery_dir).as_posix() for path in gallery_dir.rglob("*")],
            distance_threshold=config.clustering.distance_threshold,
            min_face_size=config.quality.min_face_size,
            margin=config.output.crop_margin,
            copy_originals=config.output.copy_originals,
            overwrite=config.output.overwrite,
            use_cache=use_cache,
        )
        person_dir = output_dir / "person_01"
        contact_dir = output_dir / "contact_sheets"
        person_dir.mkdir(parents=True)
        contact_dir.mkdir()
        (person_dir / "landscape_face0.jpg").write_bytes(b"crop")
        (contact_dir / "person_01.png").write_bytes(b"sheet")
        report = {
            "n_persons": 1,
            "n_faces": 1,
            "folders": {
                "person_01": {
                    "label": 0,
                    "face_count": 1,
                    "crops": [
                        {
                            "filename": "landscape_face0.jpg",
                            "source_image": str(gallery_dir / "nested" / "landscape.png"),
                            "face_index": 0,
                        }
                    ],
                    "originals": [],
                },
                "unknown": {
                    "label": -1,
                    "face_count": 0,
                    "crops": [],
                    "originals": [],
                },
            },
        }
        (output_dir / "report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return np.asarray([0], dtype=np.int32)

    monkeypatch.setattr(services, "run_cluster", fake_run_cluster)
    report, contact_sheets, archive = services.run_cluster_uploads(
        [("nested/landscape.png", uploaded_bytes)],
        load_config(),
        0.45,
        60,
        0.3,
    )

    assert set(calls["files"]) == {"nested", "nested/landscape.png"}
    assert calls["distance_threshold"] == 0.45
    assert calls["min_face_size"] == 60
    assert calls["margin"] == pytest.approx(0.3)
    assert calls["copy_originals"] is False
    assert calls["overwrite"] is True
    assert calls["use_cache"] is False
    assert report["folders"]["person_01"]["crops"][0]["source_image"] == "nested/landscape.png"
    assert contact_sheets == {"person_01": b"sheet"}
    with ZipFile(BytesIO(archive)) as zip_file:
        assert "person_01/landscape_face0.jpg" in zip_file.namelist()
        assert "contact_sheets/person_01.png" in zip_file.namelist()
        assert "report.json" in zip_file.namelist()


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

    monkeypatch.setattr(services, "run", fake_run)
    config = load_config()
    uploaded_bytes = b"uploaded image bytes"

    matches, grid_bytes, archive_bytes = services.run_search(
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


def test_streamlit_dashboard_renders_both_modes_and_requires_upload(
    tmp_path: Path,
) -> None:
    """Les deux modes s'affichent et le clustering exige un dossier importé."""
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from streamlit.testing.v1 import AppTest\n"
        "app = AppTest.from_file(sys.argv[1], default_timeout=30).run()\n"
        "assert not app.exception\n"
        "assert [item.value for item in app.title] == ['TrombiAI']\n"
        "assert 'Trie tes photos par personne, automatiquement' in [item.value for item in app.caption]\n"
        "assert [item.value for item in app.header] == ['Tri des visages par personne']\n"
        "assert [item.label for item in app.slider] == [\n"
        "    'Distance cosinus', 'Taille minimale du visage (pixels)', 'Marge autour des crops'\n"
        "]\n"
        "assert app.get('file_uploader')[0].label == 'Dossier de référence à classer'\n"
        "app.button[0].click().run()\n"
        "assert not app.exception\n"
        "assert [item.value for item in app.error] == [\n"
        "    'Importez un dossier contenant au moins une image.'\n"
        "]\n"
        "app.radio[0].set_value('Recherche historique').run()\n"
        "assert not app.exception\n"
        "assert [item.value for item in app.header] == [\n"
        "    'Recherche faciale dans une galerie locale'\n"
        "]\n"
        "assert [item.label for item in app.slider] == ['Seuil de similarité cosinus']\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(app_path)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr


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
