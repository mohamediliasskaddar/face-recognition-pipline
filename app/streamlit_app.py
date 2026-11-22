"""Entrée du tableau de bord Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
BANNER_PATH = APP_DIR / "public" / "banner.png"
STYLES_PATH = APP_DIR / "styles.css"
for search_path in (str(APP_DIR), str(PROJECT_ROOT)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from ui import components, services, state

APP_NAME = "TrombiAI"
APP_VERSION = "2.0"
APP_SLOGAN = "Trie tes photos par personne, automatiquement"


def setup_page() -> None:
    """Configure la page Streamlit."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="👥",
        layout="wide",
    )


def render_header(banner_path: Path | None = None) -> None:
    """Affiche la bannière, le nom de marque et le slogan."""
    styles = STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
    resolved_banner = BANNER_PATH if banner_path is None else banner_path
    if resolved_banner.is_file():
        st.image(
            str(resolved_banner),
            alt="Bannière TrombiAI, outil de tri facial avec Python et InsightFace",
            width="stretch",
        )
    st.title(APP_NAME)
    st.caption(APP_SLOGAN)


def render_footer() -> None:
    """Affiche les informations de version et de module."""
    st.caption(f"{APP_NAME} · v{APP_VERSION} · Module Computer Vision · Traitement local")


def main() -> None:
    """Assemble l'interface et le mode de traitement sélectionné."""
    setup_page()
    state.init_state()
    render_header()
    config = services.load_config()
    mode = components.render_sidebar()
    components.render_actions(mode, config)
    if state.has_results(mode):
        components.render_stats(mode)
        components.render_results(mode)
        components.render_downloads(mode)
    components.render_about(mode)
    render_footer()


if __name__ == "__main__":
    main()
