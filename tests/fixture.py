"""Tiny synthetic dataset in Yambda's format, placed at the same absolute timestamps as the
real split so config.toml (and every service) works on it unchanged. Used by the tests and CI.

    python -m tests.fixture data/raw
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

LAST_TS = 26_000_000
DAYS = 20


def generate(raw_dir: Path, n_users: int = 60, n_items: int = 400, n_artists: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    artist = np.arange(n_items) % n_artists
    centroids = rng.normal(size=(n_artists, 128))
    emb = centroids[artist] + 0.3 * rng.normal(size=(n_items, 128))
    item_ids = np.arange(n_items) * 7 + 100  # non-contiguous ids, like the real data
    cold = np.arange(n_items - 20, n_items)  # only appear in the last two days
    item_pop = 1.0 / np.arange(1, n_items + 1) ** 0.8

    rows = []
    start = LAST_TS - DAYS * 86_400
    for u in range(n_users):
        liked = rng.choice(n_artists, 3, replace=False)
        for _ in range(rng.integers(8, 20)):
            t = int(rng.integers(start, LAST_TS - 3_600)) // 5 * 5
            for _ in range(rng.integers(4, 25)):
                warm = t < LAST_TS - 2 * 86_400
                pool = np.flatnonzero(np.isin(artist, liked)) if rng.random() < 0.75 else np.arange(n_items)
                if warm:
                    pool = np.setdiff1d(pool, cold)
                p = item_pop[pool] / item_pop[pool].sum()
                i = int(rng.choice(pool, p=p))
                ratio = 100 if artist[i] in liked else int(rng.integers(5, 100))
                organic = int(rng.random() < 0.5)
                rows.append((u + 1, t, int(item_ids[i]), organic, ratio, 200, "listen"))
                if rng.random() < 0.05:
                    rows.append((u + 1, t, int(item_ids[i]), organic, None, None, "like"))
                # occasionally two events land in the same 5-second bin, as in the real data
                t += 0 if rng.random() < 0.1 else int(rng.integers(36, 50)) * 5
                if t >= LAST_TS:
                    break

    events = pl.DataFrame(
        rows,
        schema={
            "uid": pl.UInt32,
            "timestamp": pl.UInt32,
            "item_id": pl.UInt32,
            "is_organic": pl.UInt8,
            "played_ratio_pct": pl.UInt16,
            "track_length_seconds": pl.UInt32,
            "event_type": pl.String,
        },
        orient="row",
    ).sort("uid", "timestamp", maintain_order=True)
    (raw_dir / "flat/50m").mkdir(parents=True, exist_ok=True)
    events.write_parquet(raw_dir / "flat/50m/multi_event.parquet")
    pl.DataFrame({"artist_id": artist.astype(np.uint32), "item_id": item_ids.astype(np.uint32)}).write_parquet(
        raw_dir / "artist_item_mapping.parquet"
    )
    has_audio = rng.random(n_items) < 0.9
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(raw_dir / "embeddings.npz", item_id=item_ids[has_audio].astype(np.uint32), embed=emb[has_audio].astype(np.float16))


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
