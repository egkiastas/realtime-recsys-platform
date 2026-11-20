"""No information from after a cutoff may reach anything computed for that cutoff."""

import numpy as np
import polars as pl

from recsys.candidates import cooccurrence, popularity
from recsys.config import Config
from recsys.features import item_snapshot, session_feature_log, user_snapshot
from recsys.two_tower import training_data


def test_split_boundaries_are_ordered(cfg: Config):
    assert cfg.train_end < cfg.val_start < cfg.val_end < cfg.test_start < cfg.last_ts
    assert cfg.val_start - cfg.train_end == cfg.gap and cfg.test_start - cfg.val_end == cfg.gap


def test_requests_and_labels_stay_inside_their_period(cfg: Config, events: pl.DataFrame):
    for split, (start, end) in {"val": (cfg.val_start, cfg.val_end), "test": (cfg.test_start, cfg.last_ts + 1)}.items():
        requests = pl.read_parquet(cfg.processed_dir / f"{split}_requests.parquet")
        labels = pl.read_parquet(cfg.processed_dir / f"{split}_labels.parquet")
        assert len(requests) > 0 and len(labels) > 0
        assert requests["request_ts"].min() >= start - 1
        assert requests["request_ts"].max() + cfg.label_window <= end

        # every label is a real >=50% listen strictly after its request and inside the window
        listens = events.filter((pl.col("event_type") == "listen") & (pl.col("played_ratio_pct") >= cfg.positive_ratio))
        joined = labels.join(requests, on="request_id").join(listens, on=["uid", "item_id"])
        ok = joined.filter(
            (pl.col("timestamp") > pl.col("request_ts")) & (pl.col("timestamp") <= pl.col("request_ts") + cfg.label_window)
        )
        assert ok["request_id"].n_unique() == labels["request_id"].n_unique()
        assert ok.select(pl.struct("request_id", "item_id").n_unique()).item() == len(labels)


def test_validation_labels_never_come_from_the_test_day(cfg: Config):
    labels = pl.read_parquet(cfg.processed_dir / "val_labels.parquet")
    requests = pl.read_parquet(cfg.processed_dir / "val_requests.parquet")
    latest_possible = requests["request_ts"].max() + cfg.label_window
    assert latest_possible < cfg.test_start
    assert len(labels.join(requests, on="request_id")) == len(labels)


def test_session_log_rows_never_postdate_requests(cfg: Config, events: pl.DataFrame, tmp_path):
    """The point-in-time join may only pick rows built from events before the request."""
    out = tmp_path / "log.parquet"
    session_feature_log(events.filter(pl.col("timestamp") >= cfg.val_start - cfg.history_days * cfg.day), cfg.val_start, cfg, out)
    log = pl.read_parquet(out)
    requests = pl.read_parquet(cfg.processed_dir / "test_requests.parquet").cast({"uid": pl.Int64})
    picked = (
        requests.sort("request_ts")
        .join_asof(log.sort("last_ts"), left_on="request_ts", right_on="last_ts", by="uid", strategy="backward")
        .drop_nulls("last_ts")
    )
    assert len(picked) > 0
    assert (picked["last_ts"] < picked["request_ts"] + 1).all()


def test_batch_features_ignore_events_after_cutoff(cfg: Config, events: pl.DataFrame):
    """Appending future events must not change anything computed for an earlier cutoff."""
    item_ids = np.sort(events["item_id"].unique().to_numpy())
    artists = pl.DataFrame({"item_id": item_ids.astype(np.uint32), "artist_id": np.zeros(len(item_ids), np.uint32)})
    past = events.filter(pl.col("timestamp") < cfg.val_start)
    future = past.tail(500).with_columns(pl.col("timestamp") + (cfg.val_start - pl.col("timestamp").min()) + 10)
    with_future = pl.concat([past, future])

    assert user_snapshot(past, cfg.val_start).sort("uid").equals(user_snapshot(with_future, cfg.val_start).sort("uid"))
    assert (
        item_snapshot(past, artists, cfg.val_start)
        .sort("item_id")
        .equals(item_snapshot(with_future, artists, cfg.val_start).sort("item_id"))
    )
    assert np.array_equal(popularity(past, item_ids, cfg.val_start, cfg), popularity(with_future, item_ids, cfg.val_start, cfg))
    for a, b in zip(
        cooccurrence(past, item_ids, cfg.val_start, cfg), cooccurrence(with_future, item_ids, cfg.val_start, cfg), strict=True
    ):
        assert np.array_equal(a, b)


def test_two_tower_targets_come_from_training_period_only(cfg: Config, events: pl.DataFrame):
    item_ids = np.sort(events["item_id"].unique().to_numpy())
    seq, user_start, targets = training_data(events, item_ids, cfg)
    train_listens = events.filter((pl.col("event_type") == "listen") & (pl.col("timestamp") < cfg.train_end))
    assert len(seq) == len(train_listens)
    assert len(targets) > 0
    # a target never appears in its own recent history (same rule as the labels)
    from recsys.two_tower import gather_history

    hist = gather_history(seq, user_start, targets, cfg.hist_len)
    assert not (hist == seq[targets, None]).any()
