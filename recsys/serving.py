"""The request path: retrieve -> rank -> re-rank. Used by the API and by offline evaluation."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np

from recsys.candidates import cooc_candidates
from recsys.ranker import ALL_FEATURES, feature_matrix
from recsys.rerank import mmr

# Single-query search and small-batch prediction are slower with thread pools spinning up.
faiss.omp_set_num_threads(1)


@dataclass
class Bundle:
    version: str
    item_ids: np.ndarray  # sorted uint32; every other array is indexed by position in it
    item_feats: np.ndarray
    artist: np.ndarray
    item_vecs: np.ndarray
    user_tower: dict
    pop: np.ndarray
    fresh: np.ndarray  # cold-start pool: new or unplayed items that have an audio vector
    cooc: tuple
    index: faiss.Index
    ranker: lgb.Booster | None
    features: list[str]
    params: dict

    def __post_init__(self):
        # contiguous copy so scoring the cold pool is one matmul, not a 45k-row gather per request
        self.fresh_vecs = np.ascontiguousarray(self.item_vecs[self.fresh])


@dataclass
class Candidates:
    idx: np.ndarray
    pop_score: np.ndarray
    cooc_score: np.ndarray
    in_tt: np.ndarray
    user_vec: np.ndarray | None
    recent_idx: np.ndarray
    sources: dict  # per-source ranked lists, kept for offline recall metrics


def user_vector(item_vecs: np.ndarray, weights: dict, hist: np.ndarray) -> np.ndarray | None:
    """Numpy version of two_tower.TwoTower.user_vec, so serving does not need torch."""
    if len(hist) == 0:
        return None
    m = item_vecs[hist].astype(np.float32).mean(0)
    h = np.maximum(m @ weights["w1"] + weights["b1"], 0) @ weights["w2"] + weights["b2"]
    u = m + h
    return u / (np.linalg.norm(u) + 1e-12)


def build_index(item_vecs: np.ndarray, nprobe: int, seed: int = 42) -> faiss.Index:
    nlist = min(4096, max(1, len(item_vecs) // 50))  # 4096 lists for the 934k-item catalogue
    quantizer = faiss.IndexFlatIP(item_vecs.shape[1])
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, item_vecs.shape[1], nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    rng = np.random.default_rng(seed)
    index.train(item_vecs[rng.choice(len(item_vecs), min(len(item_vecs), 50 * nlist), replace=False)])
    index.add(item_vecs)
    index.nprobe = nprobe
    return index


def save_bundle(b: Bundle, path: Path) -> None:
    arrays = path / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    named = {
        "item_ids": b.item_ids,
        "item_feats": b.item_feats,
        "artist": b.artist,
        "item_vecs": b.item_vecs,
        "pop": b.pop,
        "fresh": b.fresh,
        "cooc_indptr": b.cooc[0],
        "cooc_nbr": b.cooc[1],
        "cooc_score": b.cooc[2],
        **{f"tower_{k}": v for k, v in b.user_tower.items()},
    }
    for name, arr in named.items():
        np.save(arrays / f"{name}.npy", np.ascontiguousarray(arr))
    faiss.write_index(b.index, str(path / "items.faiss"))
    if b.ranker is not None:
        b.ranker.save_model(str(path / "ranker.txt"))
    (path / "meta.json").write_text(json.dumps({"version": b.version, "features": b.features, "params": b.params}, indent=2))


def load_bundle(path: Path, version: str | None = None) -> Bundle:
    """Arrays are memory-mapped, so several API worker processes share one copy in the page cache."""
    path = Path(path)
    a = {f.stem: np.load(f, mmap_mode="r") for f in (path / "arrays").glob("*.npy")}
    meta = json.loads((path / "meta.json").read_text())
    index = faiss.read_index(str(path / "items.faiss"))
    index.nprobe = meta["params"]["nprobe"]
    ranker_file = path / "ranker.txt"
    return Bundle(
        version=version or meta["version"],
        item_ids=a["item_ids"],
        item_feats=a["item_feats"],
        artist=a["artist"],
        item_vecs=a["item_vecs"],
        user_tower={k.removeprefix("tower_"): np.asarray(v) for k, v in a.items() if k.startswith("tower_")},
        pop=a["pop"],
        fresh=a["fresh"],
        cooc=(a["cooc_indptr"], a["cooc_nbr"], a["cooc_score"]),
        index=index,
        ranker=lgb.Booster(model_file=str(ranker_file)) if ranker_file.exists() else None,
        features=meta["features"],
        params=meta["params"],
    )


def to_index(b: Bundle, item_ids: list[int]) -> np.ndarray:
    ids = np.asarray(item_ids, dtype=np.int64)
    pos = np.searchsorted(b.item_ids, ids).clip(max=len(b.item_ids) - 1)
    return pos[b.item_ids[pos] == ids].astype(np.int64)


def retrieve(b: Bundle, session: dict) -> Candidates:
    p = b.params
    recent = to_index(b, session["last_items"])
    recent_set = np.unique(recent)

    pop = b.pop[~np.isin(b.pop, recent_set)]
    cooc_idx, cooc_s = cooc_candidates(*b.cooc, recent, p["cooc_seeds"], p["cooc_n"])
    cooc_keep = ~np.isin(cooc_idx, recent_set)
    cooc_idx, cooc_s = cooc_idx[cooc_keep], cooc_s[cooc_keep]

    user_vec = user_vector(b.item_vecs, b.user_tower, recent[: p["hist_len"]])
    tt_idx = fresh_idx = np.empty(0, np.int64)
    if user_vec is not None:
        _, found = b.index.search(user_vec[None, :].astype(np.float32), p["tt_n"] + len(recent_set))
        found = found[0][found[0] >= 0]
        tt_idx = found[~np.isin(found, recent_set)][: p["tt_n"]]
        # Audio-only vectors rarely reach the global top-N next to warm items, so the cold
        # pool is searched separately (brute force; it is a few tens of thousands of items).
        fresh_scores = b.fresh_vecs @ user_vec
        n = min(p["fresh_n"], len(fresh_scores))
        top = np.argpartition(-fresh_scores, n - 1)[:n] if n else np.empty(0, np.int64)
        fresh_idx = b.fresh[top[np.argsort(-fresh_scores[top])]]

    idx = np.unique(np.concatenate([pop, cooc_idx, tt_idx, fresh_idx]).astype(np.int64))
    pop_score = np.zeros(len(idx), np.float32)
    pop_score[np.searchsorted(idx, pop)] = 1.0 / (1.0 + np.arange(len(pop)))
    cooc_score = np.zeros(len(idx), np.float32)
    cooc_score[np.searchsorted(idx, cooc_idx)] = cooc_s
    in_tt = np.zeros(len(idx), np.int8)
    in_tt[np.searchsorted(idx, tt_idx)] = 1
    in_tt[np.searchsorted(idx, fresh_idx)] = 1
    sources = {"popular": pop, "cooc": cooc_idx, "two_tower": tt_idx, "fresh": fresh_idx}
    return Candidates(idx, pop_score, cooc_score, in_tt, user_vec, recent, sources)


def rank(b: Bundle, cand: Candidates, session: dict, user: dict) -> np.ndarray:
    X = feature_matrix(b, cand, session, user)
    cols = [ALL_FEATURES.index(f) for f in b.features]
    return b.ranker.predict(X[:, cols], num_threads=1)


def recommend(b: Bundle, session: dict, user: dict, k: int, rerank: bool = True) -> dict:
    """Full pipeline for one request. Returns item ids plus per-stage timings in ms."""
    timings = {}
    t0 = time.perf_counter()
    cand = retrieve(b, session)
    t1 = time.perf_counter()
    scores = rank(b, cand, session, user)
    t2 = time.perf_counter()
    order = np.argsort(-scores, kind="stable")
    if rerank:
        pool = order[: b.params["pool"]]
        chosen = pool[mmr(scores[pool], b.item_vecs[cand.idx[pool]], k, b.params["mmr_lambda"])]
    else:
        chosen = order[:k]
    t3 = time.perf_counter()
    timings.update(retrieve=(t1 - t0) * 1e3, rank=(t2 - t1) * 1e3, rerank=(t3 - t2) * 1e3)
    return {
        "items": b.item_ids[cand.idx[chosen]].tolist(),
        "scores": scores[chosen].tolist(),
        "timings": timings,
        "candidates": cand,
        "ranked": cand.idx[order],
    }


def popular_fallback(b: Bundle, k: int) -> list[int]:
    return b.item_ids[b.pop[:k]].tolist()
