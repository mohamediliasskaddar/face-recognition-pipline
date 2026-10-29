"""Expansion de requête et regroupement des images par identité."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from src.config import AppConfig
from src.matching import cosine_similarity_matrix, match_images
from src.types import FaceRecord, MatchResult

QUERY_EXPANSION_MARGIN = 0.05


def _normalized_embedding(
    embedding: NDArray[np.floating],
    *,
    name: str,
) -> NDArray[np.float32]:
    """Valide et normalise un embedding."""
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} doit être un vecteur non vide.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} doit contenir uniquement des valeurs finies.")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError(f"{name} doit avoir une norme non nulle.")
    return (vector / norm).astype(np.float32)


def _record_embeddings(records: list[FaceRecord]) -> NDArray[np.float32]:
    """Valide et empile les embeddings de la galerie."""
    if not records:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = [
        _normalized_embedding(record.embedding, name=f"embedding galerie {index}")
        for index, record in enumerate(records)
    ]
    dimensions = {embedding.size for embedding in embeddings}
    if len(dimensions) != 1:
        raise ValueError("Tous les embeddings de galerie doivent avoir la même dimension.")
    return np.stack(embeddings)


def _cluster_embeddings(
    embeddings: NDArray[np.float32],
    distance_threshold: float,
) -> NDArray[np.int32]:
    """Regroupe les embeddings par distance cosinus moyenne."""
    if not np.isfinite(distance_threshold) or distance_threshold < 0:
        raise ValueError("distance_threshold doit être un nombre fini positif ou nul.")
    if embeddings.ndim != 2:
        raise ValueError("Les embeddings à regrouper doivent former une matrice (N, D).")
    if embeddings.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    if embeddings.shape[0] == 1:
        return np.zeros(1, dtype=np.int32)

    labels = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    ).fit_predict(embeddings)
    return np.asarray(labels, dtype=np.int32)


def expand_query(
    query_emb: NDArray[np.floating],
    top_matches: list[MatchResult],
    records: list[FaceRecord],
    k: int,
) -> NDArray[np.float32]:
    """Moyenne la requête avec les embeddings des k meilleures images retenues.

    ``top_matches`` doit contenir les correspondances les plus sûres, déjà triées
    par score décroissant. Les embeddings correspondants sont retrouvés par chemin
    et index de visage dans ``records``.

    Args:
        query_emb: Embedding de requête de forme `(D,)`.
        top_matches: Résultats classés utilisés pour enrichir la requête.
        records: Enregistrements ayant servi à calculer les résultats.
        k: Nombre maximal d'embeddings de résultats à inclure.

    Returns:
        Moyenne normalisée L2 en `float32`; retourne la requête normalisée si
        aucun match n'est fourni.

    Raises:
        ValueError: Si `k`, les embeddings ou un index de visage sont invalides.
    """
    query = _normalized_embedding(query_emb, name="query_emb")
    if k < 0:
        raise ValueError("k ne peut pas être négatif.")
    if k == 0 or not top_matches:
        return query

    records_by_image: dict[str, list[FaceRecord]] = {}
    for record in records:
        records_by_image.setdefault(record.image_path, []).append(record)

    selected_embeddings = [query]
    for match in top_matches[:k]:
        image_records = records_by_image.get(match.image_path)
        if image_records is None or not 0 <= match.matched_face_index < len(image_records):
            raise ValueError(
                f"Impossible de retrouver le visage {match.matched_face_index} "
                f"de '{match.image_path}' dans les enregistrements."
            )
        matched_embedding = _normalized_embedding(
            image_records[match.matched_face_index].embedding,
            name=f"embedding de '{match.image_path}'",
        )
        if matched_embedding.size != query.size:
            raise ValueError(
                "Les embeddings de requête et de galerie doivent avoir la même dimension."
            )
        selected_embeddings.append(matched_embedding)

    mean_embedding = np.mean(np.stack(selected_embeddings), axis=0)
    return _normalized_embedding(mean_embedding, name="embedding étendu")


def cluster_gallery(
    records: list[FaceRecord],
    distance_threshold: float,
) -> NDArray[np.int32]:
    """Retourne une étiquette de cluster pour chaque visage de la galerie.

    Le clustering emploie la distance cosinus et le linkage moyen. Une galerie
    vide retourne un tableau vide; un seul visage forme un cluster unique.
    """
    if not records:
        _cluster_embeddings(np.empty((0, 0), dtype=np.float32), distance_threshold)
        return np.empty(0, dtype=np.int32)
    return _cluster_embeddings(_record_embeddings(records), distance_threshold)


def resolve_identity(
    query_emb: NDArray[np.floating],
    records: list[FaceRecord],
    cfg: AppConfig,
) -> list[MatchResult]:
    """Combine le seuil initial, l'expansion et le clustering selon la config.

    Les stratégies optionnelles ajoutent des candidats sans supprimer les
    résultats retenus par la comparaison initiale. Le score reste la similarité
    cosinus utilisée par la stratégie qui a ajouté le meilleur candidat.

    Args:
        query_emb: Embedding unique de la requête `(D,)`.
        records: Visages indexés de la galerie.
        cfg: Configuration des seuils et stratégies d'identité.

    Returns:
        Images candidates triées par score décroissant.
    """
    query = _normalized_embedding(query_emb, name="query_emb")
    threshold = cfg.matching.threshold
    base_matches = match_images(query, records, threshold)
    matches_by_image = {match.image_path: match for match in base_matches}

    if cfg.matching.use_query_expansion:
        safe_matches = [
            match for match in base_matches if match.best_score > threshold + QUERY_EXPANSION_MARGIN
        ]
        if safe_matches:
            expanded_query = expand_query(
                query,
                safe_matches,
                records,
                cfg.matching.expansion_top_k,
            )
            for match in match_images(expanded_query, records, threshold):
                current = matches_by_image.get(match.image_path)
                if current is None or match.best_score > current.best_score:
                    matches_by_image[match.image_path] = match

    if cfg.matching.use_gallery_clustering and records:
        gallery_embeddings = _record_embeddings(records)
        query_matrix = _normalized_embedding(query, name="query_emb")[np.newaxis, :]
        if query_matrix.shape[1] != gallery_embeddings.shape[1]:
            raise ValueError(
                "Les embeddings de requête et de galerie doivent avoir la même dimension."
            )
        all_embeddings = np.concatenate((query_matrix, gallery_embeddings), axis=0)
        labels = _cluster_embeddings(
            all_embeddings,
            cfg.matching.cluster_distance_threshold,
        )
        query_label = labels[0]
        cluster_similarities = cosine_similarity_matrix(query, gallery_embeddings)[0]
        best_cluster_match: dict[str, MatchResult] = {}
        face_indices: dict[str, int] = {}
        for record_index, (record, score) in enumerate(
            zip(records, cluster_similarities, strict=True)
        ):
            face_index = face_indices.get(record.image_path, 0)
            face_indices[record.image_path] = face_index + 1
            if labels[record_index + 1] != query_label:
                continue
            match = MatchResult(record.image_path, float(score), face_index)
            current = best_cluster_match.get(record.image_path)
            if current is None or match.best_score > current.best_score:
                best_cluster_match[record.image_path] = match

        for match in best_cluster_match.values():
            current = matches_by_image.get(match.image_path)
            if current is None or match.best_score > current.best_score:
                matches_by_image[match.image_path] = match

    results = list(matches_by_image.values())
    results.sort(key=lambda result: result.best_score, reverse=True)
    return results
