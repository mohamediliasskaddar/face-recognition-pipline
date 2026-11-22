"""Accès centralisé à l'état de session de l'interface."""

from __future__ import annotations

import streamlit as st

RESULT_STATE_KEY = "face_search_result"
ERROR_STATE_KEY = "face_search_error"
CLUSTER_RESULT_STATE_KEY = "face_cluster_result"
CLUSTER_ERROR_STATE_KEY = "face_cluster_error"

DEFAULTS: dict[str, object] = {
    RESULT_STATE_KEY: None,
    ERROR_STATE_KEY: None,
    CLUSTER_RESULT_STATE_KEY: None,
    CLUSTER_ERROR_STATE_KEY: None,
}

_WIDGET_KEYS = (
    "face_pipeline_mode",
    "face_labels_upload",
    "face_reference_upload",
    "face_cluster_reference_upload",
)


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


def reset_all() -> None:
    """Efface les résultats, erreurs et sélections de l'interface."""
    for key in (*DEFAULTS, *_WIDGET_KEYS):
        pop(key)
