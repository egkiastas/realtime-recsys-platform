"""Download Yambda-50M, apply the global temporal split, build evaluation requests."""

import argparse
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pyarrow.compute as pc
import pyarrow.parquet as pq

from recsys.config import Config, load_config

REPO = "yandex/yambda"
REVISION = "4dcfdd370180347ac75f6ece07d76cac196e3da4"
# Yambda timestamps are seconds from an undisclosed origin. Feast needs real datetimes,
# so every timestamp is shifted onto this arbitrary epoch.
EPOCH = datetime(2024, 1, 1, tzinfo=UTC)


def to_datetime(ts: int) -> datetime:
    return EPOCH + timedelta(seconds=int(ts))


def download(cfg: Config) -> None:
    from huggingface_hub import hf_hub_download

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ["flat/50m/multi_event.parquet", "artist_item_mapping.parquet"]:
        hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=cfg.raw_dir)
        print("downloaded", name)
    download_embeddings(cfg)


def download_embeddings(cfg: Config) -> None:
    """Stream the 13.8 GB embeddings file and keep only items present in the 50M subset.

    Only the `embed` column is fetched (HTTP range reads per row group); vectors are
    L2-normalised here and stored as float16. Each row group is saved separately so an
    interrupted download resumes where it stopped.
    """
    from huggingface_hub import HfFileSystem

    out = cfg.raw_dir / "embeddings.npz"
    if out.exists():
        return
    items = pl.scan_parquet(cfg.raw_dir / "flat/50m/multi_event.parquet").select("item_id").unique().collect()
    wanted = items["item_id"].to_arrow()

    parts_dir = cfg.raw_dir / "embedding_parts"
    parts_dir.mkdir(exist_ok=True)
    fs = HfFileSystem()
    pf = pq.ParquetFile(fs.open(f"datasets/{REPO}@{REVISION}/embeddings.parquet", block_size=64 << 20))
    for rg in range(pf.metadata.num_row_groups):
        part = parts_dir / f"rg{rg:02d}.npz"
        if part.exists():
            continue
        for attempt in range(5):
            try:
                t = pf.read_row_group(rg, columns=["item_id", "embed"])
                break
            except Exception as e:  # network hiccups on a 6 GB range download are expected
                print(f"row group {rg} attempt {attempt} failed: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"row group {rg} failed 5 times")
        t = t.filter(pc.is_in(t["item_id"], value_set=wanted))
        ids = t["item_id"].to_numpy()
        emb = pc.list_flatten(t["embed"]).to_numpy().reshape(len(ids), -1)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        np.savez(part, item_id=ids, embed=emb.astype(np.float16))
        print(f"row group {rg + 1}/{pf.metadata.num_row_groups}: kept {len(ids)}")

    parts = [np.load(p) for p in sorted(parts_dir.glob("rg*.npz"))]
    np.savez(
        out,
        item_id=np.concatenate([p["item_id"] for p in parts]),
        embed=np.concatenate([p["embed"] for p in parts]),
    )
    for p in parts_dir.glob("rg*.npz"):
        p.unlink()
    parts_dir.rmdir()


def load_events(cfg: Config) -> pl.DataFrame:
    """All events, stably sorted by timestamp (per-user order within a 5 s bin is kept)."""
    return (
        pl.read_parquet(cfg.raw_dir / "flat/50m/multi_event.parquet")
        .with_columns(pl.col("event_type").cast(pl.String))
        .sort("timestamp", maintain_order=True)
    )


def load_artists(cfg: Config) -> pl.DataFrame:
    # a handful of tracks list several artists; keep the first
    return pl.read_parquet(cfg.raw_dir / "artist_item_mapping.parquet").unique("item_id", keep="first")


def make_requests(events: pl.DataFrame, start: int, end: int, cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sample recommendation requests inside [start, end) and their positives.

    A request is placed one second before a randomly chosen listen of the user, so features
    see only strictly earlier events. Positives are tracks played >= 50% in the following
    `label_window` seconds, never past `end`, so labels cannot leak across the split.
    Recently played tracks are removed later, once the request's session features are known.
    """
    listens = events.filter(pl.col("event_type") == "listen")
    period = listens.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end - cfg.label_window))
    requests = (
        period.select("uid", "timestamp")
        .unique(maintain_order=True)
        .filter(pl.int_range(pl.len()).shuffle(seed=cfg.seed).over("uid") < cfg.requests_per_user)
        .sort("uid", "timestamp")
        .with_row_index("request_id")
        .select("request_id", "uid", (pl.col("timestamp") - 1).alias("request_ts"))
    )
    positives = listens.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") < end) & (pl.col("played_ratio_pct") >= cfg.positive_ratio)
    ).select("uid", "timestamp", "item_id", "is_organic")
    labels = (
        requests.join(positives, on="uid")
        .filter((pl.col("timestamp") > pl.col("request_ts")) & (pl.col("timestamp") <= pl.col("request_ts") + cfg.label_window))
        .sort("request_id", "timestamp")
        .unique(["request_id", "item_id"], keep="first", maintain_order=True)
        .select("request_id", "item_id", "is_organic")
    )
    return requests, labels


def prepare(cfg: Config) -> None:
    """Write the split-specific files everything downstream reads."""
    out = cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    events = load_events(cfg)
    for name, (start, end) in {"val": (cfg.val_start, cfg.val_end), "test": (cfg.test_start, cfg.last_ts + 1)}.items():
        requests, labels = make_requests(events, start, end, cfg)
        requests.write_parquet(out / f"{name}_requests.parquet")
        labels.write_parquet(out / f"{name}_labels.parquet")
        print(f"{name}: {len(requests)} requests, {labels['request_id'].n_unique()} with positives")

    # The stream bootstraps from the same window the offline session-feature log starts at.
    bootstrap_from = cfg.val_start - cfg.history_days * cfg.day
    cols = ["uid", "timestamp", "item_id", "event_type", "played_ratio_pct", "is_organic"]
    recent = events.filter(pl.col("timestamp") >= bootstrap_from).select(cols)
    recent.filter(pl.col("timestamp") < cfg.test_start).write_parquet(out / "bootstrap_events.parquet")
    recent.filter(pl.col("timestamp") >= cfg.test_start).write_parquet(out / "replay_events.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["download", "prepare"])
    args = ap.parse_args()
    {"download": download, "prepare": prepare}[args.command](load_config())
