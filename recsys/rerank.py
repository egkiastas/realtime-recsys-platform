import numpy as np


def mmr(relevance: np.ndarray, vecs: np.ndarray, k: int, lam: float) -> np.ndarray:
    """Maximal marginal relevance over a candidate pool; returns positions into the pool.

    Relevance is min-max scaled so `lam` trades off on a fixed scale regardless of the
    ranker's raw score range. Vectors are assumed L2-normalised (cosine similarity).
    """
    n = len(relevance)
    k = min(k, n)
    if n == 0:
        return np.empty(0, np.int64)
    span = relevance.max() - relevance.min()
    rel = (relevance - relevance.min()) / span if span > 0 else np.ones(n)
    sim = vecs @ vecs.T
    chosen = [int(np.argmax(rel))]
    max_sim = sim[chosen[0]].copy()
    available = np.ones(n, dtype=bool)
    available[chosen[0]] = False
    for _ in range(k - 1):
        score = np.where(available, lam * rel - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(score))
        chosen.append(j)
        available[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    return np.asarray(chosen)
