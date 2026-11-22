"""Composants Streamlit du tableau de bord de reconnaissance faciale."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from ui import services, state


def render_uploader(label: str, **kwargs: object) -> object:
    """Affiche un importeur de fichiers Streamlit."""
    return st.file_uploader(label, **kwargs)


def _render_cluster_dashboard(config: services.AppConfig) -> None:
    """Affiche le formulaire d'import de dossier et les résultats de clustering."""
    st.header("Tri des visages par personne")
    st.write(
        "Importez un dossier d’images. Les visages détectés seront regroupés "
        "et les crops resteront disponibles dans cette session."
    )
    with st.sidebar:
        st.subheader("Paramètres du clustering")
        distance_threshold = st.slider(
            "Distance cosinus",
            min_value=0.0,
            max_value=2.0,
            value=float(config.clustering.distance_threshold),
            step=0.01,
            help="Une distance plus faible sépare davantage les groupes.",
        )
        min_face_size = st.slider(
            "Taille minimale du visage (pixels)",
            min_value=10,
            max_value=200,
            value=int(config.quality.min_face_size),
            step=5,
        )
        margin = st.slider(
            "Marge autour des crops",
            min_value=0.0,
            max_value=0.8,
            value=float(config.output.crop_margin),
            step=0.05,
        )

    uploaded_files = render_uploader(
        "Dossier de référence à classer",
        type=services.SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files="directory",
        key="face_cluster_reference_upload",
        help=(
            "Choisissez un dossier contenant des images JPG, PNG, BMP ou WebP. "
            "Les sous-dossiers sont conservés."
        ),
    )
    st.caption("Limites de traitement : 20 Mio par image et 250 Mio pour le dossier.")
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} fichier(s) sélectionné(s).")
    if st.button("Détecter et regrouper les visages", type="primary", width="stretch"):
        state.pop(state.CLUSTER_RESULT_STATE_KEY)
        state.pop(state.CLUSTER_ERROR_STATE_KEY)
        if not uploaded_files:
            st.error("Importez un dossier contenant au moins une image.")
        else:
            images = [
                (uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files
            ]
            try:
                with st.spinner(
                    f"Analyse de {len(images)} image(s)… "
                    "La première exécution peut charger les modèles."
                ):
                    report, contact_sheets, archive = services.run_cluster_uploads(
                        images,
                        config,
                        distance_threshold,
                        min_face_size,
                        margin,
                    )
                state.set_value(
                    state.CLUSTER_RESULT_STATE_KEY,
                    {
                        "report": report,
                        "contact_sheets": contact_sheets,
                        "archive": archive,
                    },
                )
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                state.set_value(state.CLUSTER_ERROR_STATE_KEY, str(error))

    error_message = state.get(state.CLUSTER_ERROR_STATE_KEY)
    if error_message:
        st.error(f"Le clustering a échoué : {error_message}")


def _render_search_dashboard(config: services.AppConfig) -> None:
    """Affiche le flux historique de recherche par image requête."""
    st.header("Recherche faciale dans une galerie locale")
    st.write(
        "Importez une image requête, choisissez votre dossier de galerie et lancez "
        "une recherche. Les fichiers restent sur cette machine."
    )
    with st.sidebar:
        st.subheader("Paramètres de recherche")
        gallery_value = st.text_input(
            "Dossier de la galerie",
            value=str(services.PROJECT_ROOT / "data" / "gallery"),
            help="Chemin local du dossier contenant les images à rechercher.",
        )
        reference_files = render_uploader(
            "Dossier de référence",
            type=services.SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files="directory",
            key="face_reference_upload",
            help=(
                "Importez directement une galerie locale. "
                "Lorsqu'il est sélectionné, il remplace le chemin ci-dessus."
            ),
        )
        threshold = st.slider(
            "Seuil de similarité cosinus",
            min_value=0.0,
            max_value=1.0,
            value=float(config.matching.threshold),
            step=0.01,
            help="Un seuil plus élevé réduit les faux positifs, mais peut manquer des images.",
        )
        use_cache = st.checkbox(
            "Réutiliser le cache des embeddings",
            value=True,
            help="Désactivez cette option pour recalculer les embeddings de la galerie.",
        )

    with st.form("face_search_form"):
        uploaded_file = render_uploader(
            "Image requête",
            type=services.SUPPORTED_UPLOAD_TYPES,
            help="Formats acceptés : JPG, PNG, BMP et WebP.",
        )
        submitted = st.form_submit_button(
            "Rechercher dans la galerie",
            type="primary",
            width="stretch",
        )

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        if services.is_valid_image(uploaded_bytes):
            st.image(uploaded_bytes, caption="Image requête", width=280)
        elif submitted:
            st.error("Le fichier importé n’est pas une image lisible.")

    if submitted:
        state.pop(state.RESULT_STATE_KEY)
        state.pop(state.ERROR_STATE_KEY)
        if uploaded_file is None:
            st.error("Importez une image avant de lancer la recherche.")
        elif not services.is_valid_image(uploaded_file.getvalue()):
            st.error("Le fichier importé n’est pas une image lisible.")
        else:
            try:
                if reference_files:
                    with TemporaryDirectory(prefix="face-reference-streamlit-") as temporary_dir:
                        gallery_dir = Path(temporary_dir) / "gallery"
                        reference_images = [
                            (reference_file.name, reference_file.getvalue())
                            for reference_file in reference_files
                        ]
                        services.write_uploaded_gallery(reference_images, gallery_dir)
                        gallery_images = services.list_images(gallery_dir)
                        if not gallery_images:
                            st.error("Le dossier importé ne contient aucune image prise en charge.")
                        else:
                            with st.spinner(
                                f"Recherche dans {len(gallery_images)} image(s)… "
                                "La première exécution peut charger les modèles."
                            ):
                                matches, grid_bytes, archive_bytes = services.run_search(
                                    uploaded_file.getvalue(),
                                    uploaded_file.name,
                                    gallery_dir,
                                    config,
                                    threshold,
                                    use_cache,
                                )
                            state.set_value(
                                state.RESULT_STATE_KEY,
                                {
                                    "matches": matches,
                                    "grid": grid_bytes,
                                    "archive": archive_bytes,
                                    "threshold": threshold,
                                },
                            )
                else:
                    gallery_dir = Path(gallery_value).expanduser().resolve()
                    if not gallery_dir.is_dir():
                        st.error(f"Le dossier de galerie n’existe pas : {gallery_dir}")
                    else:
                        gallery_images = services.list_images(gallery_dir)
                        if not gallery_images:
                            st.error(
                                "Le dossier sélectionné ne contient aucune image prise en charge."
                            )
                        else:
                            with st.spinner(
                                f"Recherche dans {len(gallery_images)} image(s)… "
                                "La première exécution peut charger les modèles."
                            ):
                                matches, grid_bytes, archive_bytes = services.run_search(
                                    uploaded_file.getvalue(),
                                    uploaded_file.name,
                                    gallery_dir,
                                    config,
                                    threshold,
                                    use_cache,
                                )
                            state.set_value(
                                state.RESULT_STATE_KEY,
                                {
                                    "matches": matches,
                                    "grid": grid_bytes,
                                    "archive": archive_bytes,
                                    "threshold": threshold,
                                },
                            )
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                state.set_value(state.ERROR_STATE_KEY, str(error))

    error_message = state.get(state.ERROR_STATE_KEY)
    if error_message:
        st.error(f"La recherche a échoué : {error_message}")


def render_sidebar() -> str:
    """Affiche les réglages partagés et retourne le mode sélectionné."""
    with st.sidebar:
        mode = st.radio(
            "Mode de traitement",
            ("Tri par personne", "Recherche historique"),
            key="face_pipeline_mode",
        )
        st.divider()
        st.subheader("Données de référence")
        uploaded_labels = render_uploader(
            "Fichier de labels",
            type=["csv"],
            key="face_labels_upload",
            help="CSV filename,face_index,person_id (historique : filename,person_id).",
        )
        if uploaded_labels is not None:
            try:
                with TemporaryDirectory(prefix="face-labels-streamlit-") as temporary_dir:
                    _, labels_source = services.resolve_labels_csv(uploaded_labels, temporary_dir)
                st.caption(labels_source)
            except (OSError, ValueError) as error:
                st.error(f"Labels invalides : {error}")
        elif services.DEFAULT_LABELS_PATH.is_file():
            st.caption(
                f"Labels par défaut : "
                f"{services.DEFAULT_LABELS_PATH.relative_to(services.PROJECT_ROOT).as_posix()}"
            )
        else:
            st.caption("Aucun labels.csv par défaut trouvé.")

        st.divider()
        if st.button("Réinitialiser l’interface", width="stretch"):
            state.reset_all()
            st.rerun()
    return mode


def render_actions(mode: str, config: services.AppConfig) -> None:
    """Affiche les contrôles du mode actif et lance son traitement."""
    if mode == "Tri par personne":
        _render_cluster_dashboard(config)
    else:
        _render_search_dashboard(config)


def render_about(mode: str) -> None:
    """Affiche les informations de traitement propres au mode sélectionné."""
    with st.expander("À propos du traitement et des données"):
        if mode == "Tri par personne":
            st.markdown(
                "- Le clustering fonctionne sur chaque visage, pas sur l’image entière.\n"
                "- Les crops, les planches de contact et le rapport peuvent être téléchargés en ZIP.\n"
                "- Les fichiers importés sont traités localement et le cache est désactivé pour "
                "ces imports temporaires."
            )
        else:
            st.markdown(
                "- Le pipeline détecte et compare les visages avec InsightFace/ArcFace.\n"
                "- L’index de la galerie peut être conservé dans `data/cache/` si le cache est activé.\n"
                "- Les résultats téléchargés incluent les images correspondantes, les annotations, "
                "la grille de résultats et le rapport JSON."
            )


def render_stats(mode: str) -> None:
    """Affiche les métriques du résultat de clustering, s'il existe."""
    if mode != "Tri par personne":
        return
    result = state.get(state.CLUSTER_RESULT_STATE_KEY)
    if result:
        report = result["report"]
        folders = report["folders"]
        metrics = st.columns(3)
        metrics[0].metric("Visages détectés", report["n_faces"])
        metrics[1].metric("Personnes regroupées", report["n_persons"])
        metrics[2].metric("Visages inconnus", folders["unknown"]["face_count"])


def render_results(mode: str) -> None:
    """Affiche les résultats du mode actif."""
    if mode == "Tri par personne":
        cluster_result = state.get(state.CLUSTER_RESULT_STATE_KEY)
        if not cluster_result:
            return
        report = cluster_result["report"]
        folders = report["folders"]
        contact_sheets = cluster_result["contact_sheets"]
        if not report["n_faces"]:
            st.info("Aucun visage exploitable n’a été détecté dans le dossier.")
        for folder_name, folder_data in folders.items():
            if not folder_data["crops"]:
                continue
            title = f"{folder_name} · {folder_data['face_count']} visage(s)"
            with st.expander(title, expanded=folder_name.startswith("person_")):
                sheet = contact_sheets.get(folder_name)
                if sheet is not None:
                    st.image(sheet, width="stretch")
                st.dataframe(
                    [
                        {
                            "Image source": crop["source_image"],
                            "Index du visage": crop["face_index"],
                            "Crop": crop["filename"],
                        }
                        for crop in folder_data["crops"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
        return

    result = state.get(state.RESULT_STATE_KEY)
    if not result:
        return
    st.subheader("Résultats")
    st.image(result["grid"], caption="Requête et meilleures correspondances", width="stretch")
    matches = result["matches"]
    if matches:
        st.success(f"{len(matches)} image(s) correspondante(s) trouvée(s).")
        columns = st.columns(3)
        for index, match in enumerate(matches):
            with columns[index % len(columns)]:
                st.markdown(f"**{match['filename']}**")
                st.caption(
                    f"Similarité cosinus : {match['score']:.3f} · "
                    f"visage : {match['matched_face_index']}"
                )
    else:
        st.info("Aucune correspondance n’a dépassé le seuil sélectionné.")


def render_downloads(mode: str) -> None:
    """Affiche le téléchargement ZIP du résultat courant."""
    if mode == "Tri par personne":
        cluster_result = state.get(state.CLUSTER_RESULT_STATE_KEY)
        if not cluster_result:
            return
        st.download_button(
            label="Télécharger les crops et rapports (ZIP)",
            data=cluster_result["archive"],
            file_name="face-clustering-results.zip",
            mime="application/zip",
            width="stretch",
        )
        return
    result = state.get(state.RESULT_STATE_KEY)
    if result:
        st.download_button(
            label="Télécharger les résultats (ZIP)",
            data=result["archive"],
            file_name="face-recognition-results.zip",
            mime="application/zip",
            width="stretch",
        )
