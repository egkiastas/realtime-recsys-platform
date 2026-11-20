"""Offline metrics over evaluation requests.

Each request record holds the positives (item indices, recently played already removed),
their is_organic flags, and the ranked lists produced by every method. Recall@K is
hits / min(|positives|, K) so a request with 3 positives can reach 1.0 at K=10.
"""

import numpy as np


def recall(ranked: np.ndarray, pos: np.ndarray, k: int) -> float:
    return len(np.intersect1d(ranked[:k], pos)) / min(len(pos), k)


def ndcg(ranked: np.ndarray, pos: np.ndarray, k: int) -> float:
    hits = np.isin(ranked[:k], pos)
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    ideal = (1 / np.log2(np.arange(2, min(len(pos), k) + 2))).sum()
    return float(dcg / ideal)


def intra_list_diversity(items: np.ndarray, item_vecs: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity of a list (item vectors are L2-normalised)."""
    v = item_vecs[items]
    sim = v @ v.T
    n = len(items)
    return float(1 - (sim.sum() - np.trace(sim)) / (n * (n - 1))) if n > 1 else 0.0


def method_table(records: list[dict], methods: list[str], item_pop: np.ndarray, item_vecs: np.ndarray, k: int = 10) -> dict:
    """Accuracy and beyond-accuracy metrics for each method's top-k.

    item_pop: listens over the last 30 days per item index (popularity reference).
    Coverage is distinct recommended items over items played in those 30 days.
    Popularity percentile is the mean percentile of recommended items among those items.
    Diversity is intra-list diversity over the two-tower item vectors.
    """
    item_pop = np.nan_to_num(item_pop)
    active = item_pop > 0
    sorted_pop = np.sort(item_pop[active])
    total = item_pop.sum()
    out = {}
    for m in methods:
        lists = [r["lists"][m][:k] for r in records]
        recs = np.concatenate(lists)
        out[m] = {
            f"recall@{k}": float(np.mean([recall(r["lists"][m], r["pos"], k) for r in records])),
            f"ndcg@{k}": float(np.mean([ndcg(r["lists"][m], r["pos"], k) for r in records])),
            "distinct": len(np.unique(recs)),
            "coverage": len(np.unique(recs)) / int(active.sum()),
            "novelty": float(np.mean(-np.log2((item_pop[recs] + 1) / total))),
            "pop_percentile": float(np.mean(np.searchsorted(sorted_pop, item_pop[recs]) / len(sorted_pop) * 100)),
            "diversity": float(np.mean([intra_list_diversity(lst, item_vecs) for lst in lists])),
        }
    pos = np.concatenate([r["pos"] for r in records])
    out["ground truth"] = {
        "novelty": float(np.mean(-np.log2((item_pop[pos] + 1) / total))),
        "pop_percentile": float(np.mean(np.searchsorted(sorted_pop, item_pop[pos]) / len(sorted_pop) * 100)),
    }
    return out


def paired_bootstrap(records: list[dict], a: str, b: str, k: int = 10, n: int = 1000, seed: int = 0) -> dict:
    """95% interval for the mean per-request difference in Recall@k between methods a and b."""
    diff = np.array([recall(r["lists"][a], r["pos"], k) - recall(r["lists"][b], r["pos"], k) for r in records])
    rng = np.random.default_rng(seed)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return {"mean": float(diff.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def candidate_recall(records: list[dict], sources: list[str], ks=(100, 200)) -> dict:
    out = {s: {f"recall@{k}": float(np.mean([recall(r["lists"][s], r["pos"], k) for r in records])) for k in ks} for s in sources}
    out["union"] = {
        "recall": float(np.mean([len(np.intersect1d(r["union"], r["pos"])) / len(r["pos"]) for r in records])),
        "mean_size": float(np.mean([len(r["union"]) for r in records])),
    }
    return out


def sliced_recall(records: list[dict], methods: list[str], cold: np.ndarray, light_users: set, k: int = 10) -> dict:
    """Recall@k restricted to a subset of positives (or of requests), per method.

    organic / recommendation-driven: positives whose listen was organic or came from the
    platform's recommender. cold items: positives with < 5 listens before the cutoff.
    light users: requests from users in the bottom quintile of history length.
    """

    def subset(r, name):
        if name == "organic":
            return r["pos"][r["organic"] == 1]
        if name == "recommendation-driven":
            return r["pos"][r["organic"] == 0]
        if name == "cold items":
            return r["pos"][cold[r["pos"]]]
        if name == "light users":
            return r["pos"] if r["uid"] in light_users else r["pos"][:0]
        return r["pos"]

    out = {}
    for name in ["all", "organic", "recommendation-driven", "cold items", "light users"]:
        subs = [(r, subset(r, name)) for r in records]
        subs = [(r, s) for r, s in subs if len(s)]
        row = {"requests": len(subs)}
        for m in methods:
            row[m] = float(np.mean([recall(r["lists"][m], s, k) for r, s in subs])) if subs else float("nan")
        row["union"] = float(np.mean([len(np.intersect1d(r["union"], s)) / len(s) for r, s in subs])) if subs else float("nan")
        out[name] = row
    return out
