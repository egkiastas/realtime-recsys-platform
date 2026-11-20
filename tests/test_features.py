"""Training-serving skew: the features the ranker is trained on (Feast point-in-time join over
the offline log) must equal what the API reads online after the stream has processed the
same events. Runs against SQLite always and against Redis when one is reachable, because
Redis has write semantics (ignore non-newer timestamps) that SQLite does not.
"""

import os
import uuid

import numpy as np
import pandas as pd
import polars as pl
import pytest
import redis
from feast import FeatureStore, RepoConfig
from feast.infra.offline_stores.dask import DaskOfflineStoreConfig
from feast.infra.online_stores.redis import RedisOnlineStoreConfig
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig

from recsys import store
from recsys.config import Config
from recsys.data import to_datetime
from recsys.features import SessionState, session_feature_log, update, user_snapshot
from services.stream.main import bootstrap, process

REDIS = os.environ.get("REDIS_TEST", "localhost:6380")


def redis_available() -> bool:
    host, port = REDIS.split(":")
    try:
        return redis.Redis(host=host, port=int(port), socket_timeout=0.2).ping()
    except redis.RedisError:
        return False


def feature_store(tmp_path, backend: str) -> FeatureStore:
    online = (
        SqliteOnlineStoreConfig(path=str(tmp_path / "online.db"))
        if backend == "sqlite"
        else RedisOnlineStoreConfig(connection_string=REDIS)
    )
    return FeatureStore(
        config=RepoConfig(
            project=f"test_{uuid.uuid4().hex[:8]}",
            registry=str(tmp_path / "registry.db"),
            provider="local",
            online_store=online,
            offline_store=DaskOfflineStoreConfig(),
            entity_key_serialization_version=3,
        )
    )


def assert_same(online: dict, offline: pd.Series, uid: int) -> None:
    for name, value in online.items():
        if name == "uid":
            continue
        other = offline[name]
        if name == "last_items":
            assert list(value or []) == list(other if other is not None else []), f"{name} differs for user {uid}"
        elif value is None:
            assert other is None or pd.isna(other), f"{name} differs for user {uid}"
        else:
            assert value == pytest.approx(float(other)), f"{name} differs for user {uid}: {value} vs {other}"


@pytest.mark.parametrize(
    "backend", ["sqlite", pytest.param("redis", marks=pytest.mark.skipif(not redis_available(), reason="no Redis"))]
)
def test_online_features_match_point_in_time_join(cfg: Config, events: pl.DataFrame, tmp_path, backend):
    data = tmp_path / "data"
    data.mkdir()
    bootstrap_from = cfg.val_start - cfg.history_days * cfg.day
    recent = events.filter(pl.col("timestamp") >= bootstrap_from)
    session_feature_log(recent, cfg.val_start, cfg, data / "session_features.parquet")
    pl.concat([user_snapshot(events, cfg.val_start), user_snapshot(events, cfg.test_start)]).write_parquet(
        data / "user_features.parquet"
    )

    fs = feature_store(tmp_path, backend)
    store.apply(fs, data)
    fs.materialize(to_datetime(cfg.val_start + 1), to_datetime(cfg.test_start + 1), feature_views=["user_stats"])

    # Online side: exactly what the stream service does, in micro-batches of 25 events.
    cutoff = cfg.test_start + 18 * 3600
    states = bootstrap(fs, recent.filter(pl.col("timestamp") < cfg.test_start))
    live = recent.filter((pl.col("timestamp") >= cfg.test_start) & (pl.col("timestamp") < cutoff)).to_dicts()
    for i in range(0, len(live), 25):
        process(fs, states, live[i : i + 25])

    users = sorted(set(states))
    entity_df = pd.DataFrame({"uid": users, "event_timestamp": pd.Timestamp(to_datetime(cutoff - 1))})
    offline = fs.get_historical_features(entity_df=entity_df, features=store.ONLINE_FEATURES).to_df().set_index("uid")
    assert len(live) > 50 and len(users) > 20
    for uid in users:
        assert_same(store.online_features(fs, uid), offline.loc[uid], uid)


def test_events_in_same_bin_are_all_applied(cfg: Config, tmp_path):
    """Yambda bins timestamps to 5 s. Without the per-bin microsecond offset the second push
    has the same timestamp as the first and Feast's Redis store silently drops it."""
    if not redis_available():
        pytest.skip("needs Redis")
    fs = feature_store(tmp_path, "redis")
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame(schema={"uid": pl.Int64}).write_parquet(data / "session_features.parquet")
    store.apply(fs, data)
    st = SessionState()
    for item in [11, 22]:
        store.push_session_rows(fs, [{"uid": 7, **update(st, cfg.test_start, item, "listen", 100, 1, cfg)}])
    assert store.online_features(fs, 7)["last_items"] == [22, 11]


def test_session_window_and_reset(cfg: Config):
    st = SessionState()
    t0 = cfg.test_start
    update(st, t0, 1, "listen", 100, 1, cfg)
    update(st, t0 + 60, 2, "listen", 10, 0, cfg)  # skip
    row = update(st, t0 + 120, 2, "like", None, 1, cfg)
    assert (row["listens_30m"], row["skips_30m"], row["likes_30m"]) == (2, 1, 1)
    assert row["skip_rate_30m"] == 0.5
    assert row["session_length"] == 3
    assert row["last_items"] == [2, 1]

    # 25 minutes later the first events are still inside the 30-minute window...
    row = update(st, t0 + 1500, 3, "listen", 100, 1, cfg)
    assert row["listens_30m"] == 3 and row["session_length"] == 4
    # ...and after a gap longer than the session window the session restarts
    row = update(st, t0 + 1500 + cfg.session_window + 5, 4, "listen", 100, 1, cfg)
    assert row["listens_30m"] == 1 and row["session_length"] == 1
    assert row["last_items"][:2] == [4, 3]


def test_last_items_is_capped(cfg: Config):
    st = SessionState()
    for i in range(cfg.last_n + 10):
        row = update(st, cfg.test_start + 5 * i, i, "listen", 100, 1, cfg)
    assert len(row["last_items"]) == cfg.last_n
    assert row["last_items"][0] == cfg.last_n + 9
    assert np.all(np.diff(row["last_items"]) == -1)
