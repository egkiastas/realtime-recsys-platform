"""Feature definitions shared by training and serving.

Session features are computed by one incremental function, `update`. The stream processor
calls it per event and pushes the result to Feast; offline, `session_feature_log` replays
the same events through the same function and writes the rows Feast uses for point-in-time
joins. There is no second implementation to drift out of sync.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from recsys.config import Config
from recsys.data import EPOCH

SESSION_FEATURES = [
    "last_items",
    "last_ts",
    "listens_30m",
    "skips_30m",
    "likes_30m",
    "skip_rate_30m",
    "organic_share_30m",
    "session_length",
]
USER_FEATURES = [
    "user_listens",
    "user_listens_30d",
    "user_pos_rate",
    "user_likes",
    "user_organic_share",
    "user_distinct_items",
    "user_active_days",
]
ITEM_FEATURES = [
    "item_listens_7d",
    "item_listens_30d",
    "item_pos_rate",
    "item_like_rate",
    "item_organic_share",
    "item_age_days",
    "track_length",
    "artist_listens_30d",
]


@dataclass
class SessionState:
    last_items: deque = field(default_factory=deque)  # most recent last
    window: deque = field(default_factory=deque)  # (ts, is_listen, is_skip, is_like, is_organic)
    last_ts: int = -1
    seq: int = 0  # index of the event inside its 5-second timestamp bin
    session_length: int = 0


def update(
    state: SessionState,
    ts: int,
    item_id: int,
    event_type: str,
    played_ratio: int | None,
    is_organic: int,
    cfg: Config,
) -> dict:
    """Apply one event to the user's state and return the feature row to store.

    Events must arrive in timestamp order per user. Several events can share a 5 s bin;
    each gets a microsecond offset because Feast's Redis store ignores a write whose
    timestamp is not strictly newer than the stored one.
    """
    if state.last_ts >= 0 and ts - state.last_ts > cfg.session_window:
        state.session_length = 0
    state.seq = state.seq + 1 if ts == state.last_ts else 0
    state.session_length += 1
    state.last_ts = ts

    is_listen = event_type == "listen"
    if is_listen:
        state.last_items.append(item_id)
        if len(state.last_items) > cfg.last_n:
            state.last_items.popleft()
    is_skip = is_listen and (played_ratio or 0) < cfg.skip_ratio
    state.window.append((ts, is_listen, is_skip, event_type == "like", is_organic))
    while state.window[0][0] <= ts - cfg.session_window:
        state.window.popleft()

    listens = sum(e[1] for e in state.window)
    skips = sum(e[2] for e in state.window)
    return {
        "event_timestamp": EPOCH + timedelta(seconds=ts, microseconds=state.seq),
        "last_items": list(reversed(state.last_items)),
        "last_ts": ts,
        "listens_30m": listens,
        "skips_30m": skips,
        "likes_30m": sum(e[3] for e in state.window),
        "skip_rate_30m": skips / listens if listens else 0.0,
        "organic_share_30m": sum(e[4] for e in state.window) / len(state.window),
        "session_length": state.session_length,
    }


SESSION_SCHEMA = pa.schema(
    [
        ("uid", pa.int64()),
        ("event_timestamp", pa.timestamp("us", tz="UTC")),
        ("last_items", pa.list_(pa.int64())),
        ("last_ts", pa.int64()),
        ("listens_30m", pa.int64()),
        ("skips_30m", pa.int64()),
        ("likes_30m", pa.int64()),
        ("skip_rate_30m", pa.float64()),
        ("organic_share_30m", pa.float64()),
        ("session_length", pa.int64()),
    ]
)

EVENT_COLUMNS = ["uid", "timestamp", "item_id", "event_type", "played_ratio_pct", "is_organic"]


def replay(events: pl.DataFrame, cfg: Config, states: dict | None = None):
    """Yield (uid, row) for each event, in timestamp order."""
    states = {} if states is None else states
    for uid, ts, item, etype, ratio, org in events.select(EVENT_COLUMNS).iter_rows():
        st = states.get(uid)
        if st is None:
            st = states[uid] = SessionState()
        yield uid, update(st, ts, item, etype, ratio, org, cfg)


def session_feature_log(events: pl.DataFrame, emit_from: int, cfg: Config, out: Path) -> None:
    """Write the offline session-feature log used for point-in-time joins.

    `events` start at the bootstrap time (same as the stream's bootstrap). Rows before
    `emit_from` are never the latest row for a request after `emit_from`, except each
    user's final pre-`emit_from` state, so only that one is kept.
    """
    events = events.sort("timestamp", maintain_order=True)
    before = events.filter(pl.col("timestamp") < emit_from)
    after = events.filter(pl.col("timestamp") >= emit_from)
    states: dict = {}
    last_rows = {}
    for uid, row in replay(before, cfg, states):
        last_rows[uid] = row
    writer = pq.ParquetWriter(out, SESSION_SCHEMA)
    buf = [{"uid": uid, **row} for uid, row in last_rows.items()]
    for uid, row in replay(after, cfg, states):
        buf.append({"uid": uid, **row})
        if len(buf) >= 200_000:
            writer.write_table(pa.Table.from_pylist(buf, SESSION_SCHEMA))
            buf = []
    writer.write_table(pa.Table.from_pylist(buf, SESSION_SCHEMA))
    writer.close()


def user_snapshot(events: pl.DataFrame, cutoff: int) -> pl.DataFrame:
    """Daily batch user features as of `cutoff`, stamped with the cutoff time."""
    hist = events.filter(pl.col("timestamp") < cutoff)
    listens = hist.filter(pl.col("event_type") == "listen")
    agg = listens.group_by("uid").agg(
        pl.len().alias("user_listens"),
        (pl.col("timestamp") >= cutoff - 30 * 86_400).sum().alias("user_listens_30d"),
        (pl.col("played_ratio_pct") >= 50).mean().alias("user_pos_rate"),
        pl.col("is_organic").mean().alias("user_organic_share"),
        pl.col("item_id").n_unique().alias("user_distinct_items"),
        (pl.col("timestamp") // 86_400).n_unique().alias("user_active_days"),
    )
    likes = hist.filter(pl.col("event_type") == "like").group_by("uid").agg(pl.len().alias("user_likes"))
    return (
        agg.join(likes, on="uid", how="left")
        .with_columns(
            pl.col("user_likes").fill_null(0),
            pl.lit(EPOCH + timedelta(seconds=cutoff)).alias("event_timestamp"),
        )
        .select(
            pl.col("uid").cast(pl.Int64),
            "event_timestamp",
            *[pl.col(c).cast(pl.Float64) for c in USER_FEATURES],
        )
    )


def item_snapshot(events: pl.DataFrame, artists: pl.DataFrame, cutoff: int) -> pl.DataFrame:
    """Item features as of `cutoff`. Served from memory in the API, not from Feast."""
    hist = events.filter(pl.col("timestamp") < cutoff)
    listens = hist.filter(pl.col("event_type") == "listen")
    day = 86_400
    agg = listens.group_by("item_id").agg(
        (pl.col("timestamp") >= cutoff - 7 * day).sum().alias("item_listens_7d"),
        (pl.col("timestamp") >= cutoff - 30 * day).sum().alias("item_listens_30d"),
        # smoothed towards the global positive rate (~0.63) so rare items are not extreme
        (((pl.col("played_ratio_pct") >= 50).sum() + 6.3) / (pl.len() + 10)).alias("item_pos_rate"),
        pl.len().alias("n_listens"),
        pl.col("is_organic").mean().alias("item_organic_share"),
        ((cutoff - pl.col("timestamp").min()) / day).alias("item_age_days"),
        pl.col("track_length_seconds").median().alias("track_length"),
    )
    likes = hist.filter(pl.col("event_type") == "like").group_by("item_id").agg(pl.len().alias("likes"))
    out = (
        agg.join(likes, on="item_id", how="left")
        .with_columns(((pl.col("likes").fill_null(0) + 0.1) / (pl.col("n_listens") + 10)).alias("item_like_rate"))
        .join(artists, on="item_id", how="left")
    )
    artist_pop = out.group_by("artist_id").agg(pl.col("item_listens_30d").sum().alias("artist_listens_30d"))
    return (
        out.join(artist_pop, on="artist_id", how="left")
        .with_columns(pl.col("artist_listens_30d").fill_null(0), pl.col("artist_id").fill_null(-1))
        .select("item_id", "artist_id", "n_listens", *[pl.col(c).cast(pl.Float32) for c in ITEM_FEATURES])
    )


def session_dict(row: dict) -> dict:
    """Normalise a session-feature row from Feast (online or offline) to plain Python types."""
    items = row.get("last_items")
    return {
        "last_items": [] if items is None else [int(i) for i in items],
        **{
            k: (0 if row.get(k) is None or (isinstance(row[k], float) and np.isnan(row[k])) else row[k])
            for k in SESSION_FEATURES[1:]
        },
    }
