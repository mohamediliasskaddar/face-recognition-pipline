"""Comparaison des embeddings et recherche de correspondances."""

import numpy as np
from numpy.typing import NDArray

from src.types import FaceRecord, MatchResult


def cosine_similarity_matrix(
    query_emb: NDArray[np.floating],
    gallery_embs: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Calcule les similarités cosinus entre une ou plusieurs requêtes et la galerie.

    Args:
        query_emb: Vecteur de requête `(D,)` ou matrice `(Q, D)`.
        gallery_embs: Matrice des embeddings galerie `(N, D)`.

    Returns:
        Matrice de similarité de forme `(Q, N)`.

    Raises:
        ValueError: Si les dimensions sont incompatibles, les valeurs non finies
            ou si un embedding a une norme nulle.
    """
    queries = np.asarray(query_emb, dtype=np.float64)
    gallery = np.asarray(gallery_embs, dtype=np.float64)
    if queries.ndim == 1:
        queries = queries[np.newaxis, :]
    if queries.ndim != 2:
        raise ValueError("query_emb doit être un vecteur (D,) ou une matrice (Q, D).")
    if gallery.ndim != 2:
        raise ValueError("gallery_embs doit être une matrice (N, D).")
    if queries.shape[1] == 0 or gallery.shape[1] == 0:
        raise ValueError("Les embeddings doivent contenir au moins une dimension.")
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError(
            "Les embeddings de requête et de galerie doivent avoir la même dimension."
        )
    if not np.isfinite(queries).all() or not np.isfinite(gallery).all():
        raise ValueError("Les embeddings doivent contenir uniquement des valeurs finies.")

    query_norms = np.linalg.norm(queries, axis=1)
    gallery_norms = np.linalg.norm(gallery, axis=1)
    if np.any(query_norms == 0) or np.any(gallery_norms == 0):
        raise ValueError("Les embeddings doivent avoir une norme non nulle.")

    similarities = (queries @ gallery.T) / (query_norms[:, np.newaxis] * gallery_norms)
    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


def match_images(
    query_emb: NDArray[np.floating],
    records: list[FaceRecord],
    threshold: float,
) -> list[MatchResult]:
    """Retourne les images dont le visage le plus similaire atteint le seuil.

    Args:
        query_emb: Embedding unique de la requête `(D,)`.
        records: Enregistrements des visages indexés dans la galerie.
        threshold: Similarité cosinus minimale, seuil inclusif.

    Returns:
        Résultats triés par score décroissant. ``matched_face_index`` est
        l'index zéro-based du visage gagnant parmi les visages de son image.

    Raises:
        ValueError: Si la requête, le seuil ou les embeddings sont invalides.
    """
    query = np.asarray(query_emb, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query_emb doit être un vecteur de forme (D,).")
    if not np.isfinite(threshold):
        raise ValueError("threshold doit être un nombre fini.")
    if not records:
        cosine_similarity_matrix(query, np.empty((0, query.size), dtype=np.float32))
        return []

    embeddings = [np.asarray(record.embedding, dtype=np.float32) for record in records]
    if any(embedding.ndim != 1 for embedding in embeddings):
        raise ValueError("Chaque embedding galerie doit être un vecteur de forme (D,).")
    if any(embedding.size != query.size for embedding in embeddings):
        raise ValueError(
            "Les embeddings de requête et de galerie doivent avoir la même dimension."
        )

    similarities = cosine_similarity_matrix(query, np.stack(embeddings))[0]
    face_indices: dict[str, int] = {}
    best_by_image: dict[str, tuple[float, int]] = {}
    for record, similarity in zip(records, similarities, strict=True):
        face_index = face_indices.get(record.image_path, 0)
        face_indices[record.image_path] = face_index + 1
        score = float(similarity)
        current_best = best_by_image.get(record.image_path)
        if current_best is None or score > current_best[0]:
            best_by_image[record.image_path] = (score, face_index)

    results = [
        MatchResult(image_path, score, face_index)
        for image_path, (score, face_index) in best_by_image.items()
        if score >= threshold
    ]
    results.sort(key=lambda result: result.best_score, reverse=True)
    return results
