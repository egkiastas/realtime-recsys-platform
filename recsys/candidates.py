"""Non-learned candidate sources: recent popularity and item-item co-occurrence.

Everything here works in item-index space (positions in the bundle's sorted item_ids).
"""

import numpy as np
import polars as pl

from recsys.config import Config


def popularity(events: pl.DataFrame, item_ids: np.ndarray, cutoff: int, cfg: Config) -> np.ndarray:
    """Top items by >=50%-played listens over the last `pop_days` before `cutoff`."""
    top = (
        events.filter(
            (pl.col("timestamp") < cutoff)
            & (pl.col("timestamp") >= cutoff - cfg.pop_days * cfg.day)
            & (pl.col("event_type") == "listen")
            & (pl.col("played_ratio_pct") >= cfg.positive_ratio)
        )
        .group_by("item_id")
        .len()
        .sort(["len", "item_id"], descending=[True, False])
        .head(cfg.pop_n)
    )
    return np.searchsorted(item_ids, top["item_id"].to_numpy()).astype(np.int32)


def cooccurrence(events: pl.DataFrame, item_ids: np.ndarray, cutoff: int, cfg: Config, days: int = 90):
    """Directed "played next" graph from positive listens within a session.

    For each listen, the next `cooc_window` positive listens of the same user within the
    session window count as co-occurrences. Scores are counts divided by sqrt(target count)
    to damp the pull towards already-popular tracks. Returns CSR arrays (indptr, nbr, score).
    """
    pos = (
        events.filter(
            (pl.col("timestamp") < cutoff)
            & (pl.col("timestamp") >= cutoff - days * cfg.day)
            & (pl.col("event_type") == "listen")
            & (pl.col("played_ratio_pct") >= cfg.positive_ratio)
        )
        .select("uid", "timestamp", "item_id")
        .sort("uid", "timestamp", maintain_order=True)
    )
    pairs = pl.concat(
        [
            pos.with_columns(
                pl.col("item_id").shift(-d).over("uid").alias("dst"),
                (pl.col("timestamp").shift(-d).over("uid") - pl.col("timestamp")).alias("dt"),
            ).filter((pl.col("dt") <= cfg.session_window) & (pl.col("dst") != pl.col("item_id")))
            for d in range(1, cfg.cooc_window + 1)
        ]
    ).select(pl.col("item_id").alias("src"), "dst")
    counts = pos.group_by("item_id").len().rename({"item_id": "dst", "len": "dst_count"})
    graph = (
        pairs.group_by("src", "dst")
        .len()
        .filter(pl.col("len") >= 2)
        .join(counts, on="dst")
        .with_columns((pl.col("len") / pl.col("dst_count").sqrt()).alias("score"))
        .sort(["src", "score", "dst"], descending=[False, True, False])
        .filter(pl.int_range(pl.len()).over("src") < cfg.cooc_topk)
    )
    src = np.searchsorted(item_ids, graph["src"].to_numpy())
    indptr = np.zeros(len(item_ids) + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    return (
        np.cumsum(indptr),
        np.searchsorted(item_ids, graph["dst"].to_numpy()).astype(np.int32),
        graph["score"].to_numpy().astype(np.float32),
    )


def cooc_candidates(indptr, nbr, score, recent: np.ndarray, n_seeds: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate neighbours of the most recent distinct items, weighting newer seeds more."""
    _, first = np.unique(recent, return_index=True)
    seeds = recent[np.sort(first)][:n_seeds]
    if len(seeds) == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    starts, ends = indptr[seeds], indptr[seeds + 1]
    lengths = ends - starts
    if lengths.sum() == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends, strict=True)])
    weights = np.repeat(1.0 / (1.0 + np.arange(len(seeds))), lengths)
    items, inv = np.unique(nbr[idx], return_inverse=True)
    agg = np.zeros(len(items), dtype=np.float32)
    np.add.at(agg, inv, score[idx] * weights)
    top = np.argsort(-agg, kind="stable")[:n]
    return items[top].astype(np.int32), agg[top]
