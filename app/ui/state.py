"""Accès centralisé à l'état de session de l'interface."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st

TEMP_ROOT = Path(tempfile.gettempdir()).resolve()
RESULT_STATE_KEY = "face_search_result"
ERROR_STATE_KEY = "face_search_error"
CLUSTER_RESULT_STATE_KEY = "face_cluster_result"
CLUSTER_ERROR_STATE_KEY = "face_cluster_error"

DEFAULTS: dict[str, object] = {
    "uploader_key": 0,
    "run_id": None,
    "workdir": None,
    RESULT_STATE_KEY: None,
    ERROR_STATE_KEY: None,
    CLUSTER_RESULT_STATE_KEY: None,
    CLUSTER_ERROR_STATE_KEY: None,
}


def init_state() -> None:
    """Initialise les valeurs applicatives de session sans écraser les résultats."""
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def get(key: str, default: object = None) -> object:
    """Retourne une valeur d'état applicative."""
    return st.session_state.get(key, default)


def set_value(key: str, value: object) -> None:
    """Enregistre une valeur d'état applicative."""
    st.session_state[key] = value


def pop(key: str) -> object | None:
    """Supprime une valeur d'état applicative si elle existe."""
    return st.session_state.pop(key, None)


def has_results(mode: str) -> bool:
    """Indique si un mode a produit des résultats."""
    result_key = CLUSTER_RESULT_STATE_KEY if mode == "Tri par personne" else RESULT_STATE_KEY
    return bool(get(result_key))


def get_uploader_key() -> int:
    """Retourne la génération courante des clés d'importeur."""
    uploader_key = get("uploader_key", 0)
    if not isinstance(uploader_key, int):
        raise TypeError("La clé de génération de l’importeur doit être un entier.")
    return uploader_key


def ensure_workdir() -> Path:
    """Retourne le dossier temporaire isolé de la session, en le créant au besoin."""
    workdir = get("workdir")
    if workdir is None:
        created_workdir = Path(tempfile.mkdtemp(prefix="trombiai_", dir=TEMP_ROOT))
        set_value("workdir", str(created_workdir))
        return created_workdir
    if not isinstance(workdir, (str, Path)):
        raise TypeError("Le dossier temporaire de session est invalide.")
    return _validate_workdir(workdir)


def new_run_dir(prefix: str) -> Path:
    """Crée un répertoire d'exécution dans le dossier temporaire de la session."""
    workdir = ensure_workdir()
    run_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=workdir))
    set_value("run_id", run_dir.name)
    return run_dir


def reset_all() -> None:
    """Efface les données temporaires et réinitialise l'état sans toucher aux modèles."""
    had_results = any(get(key) for key in (RESULT_STATE_KEY, CLUSTER_RESULT_STATE_KEY))
    workdir = get("workdir")
    if workdir is not None:
        if not isinstance(workdir, (str, Path)):
            raise TypeError("Le dossier temporaire de session est invalide.")
        safe_workdir = _validate_workdir(workdir)
        if safe_workdir.exists():
            shutil.rmtree(safe_workdir)

    next_uploader_key = get_uploader_key() + 1
    for key, value in DEFAULTS.items():
        set_value(key, next_uploader_key if key == "uploader_key" else value)
    if had_results:
        st.toast("Session réinitialisée")


def _validate_workdir(workdir: str | Path) -> Path:
    """Refuse toute suppression en dehors du dossier temporaire TrombiAI."""
    resolved_workdir = Path(workdir).expanduser().resolve()
    if (
        resolved_workdir.parent != TEMP_ROOT
        or not resolved_workdir.name.startswith("trombiai_")
    ):
        raise ValueError("Refus de supprimer un dossier hors de la session temporaire TrombiAI.")
    return resolved_workdir
