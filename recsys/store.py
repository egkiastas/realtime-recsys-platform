"""Feast objects and the few calls the rest of the code makes against the feature store."""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, PushSource, ValueType
from feast.data_source import PushMode
from feast.types import Array, Float64, Int64

from recsys.config import ROOT, Config, load_config
from recsys.data import to_datetime
from recsys.features import SESSION_FEATURES, USER_FEATURES

REPO = ROOT / "infra" / "feast"
PUSH_SOURCE = "session_push"
ONLINE_FEATURES = [f"session:{f}" for f in SESSION_FEATURES] + [f"user_stats:{f}" for f in USER_FEATURES]


def definitions(data_dir: Path) -> list:
    user = Entity(name="user", join_keys=["uid"], value_type=ValueType.INT64)
    session_push = PushSource(
        name=PUSH_SOURCE,
        batch_source=FileSource(
            name="session_log", path=str(data_dir / "session_features.parquet"), timestamp_field="event_timestamp"
        ),
    )
    session = FeatureView(
        name="session",
        entities=[user],
        # The online store never expires rows, so the offline join must not either,
        # otherwise an inactive user gets nulls offline and a stale state online.
        ttl=timedelta(0),
        schema=[
            Field(name="last_items", dtype=Array(Int64)),
            Field(name="last_ts", dtype=Int64),
            Field(name="listens_30m", dtype=Int64),
            Field(name="skips_30m", dtype=Int64),
            Field(name="likes_30m", dtype=Int64),
            Field(name="skip_rate_30m", dtype=Float64),
            Field(name="organic_share_30m", dtype=Float64),
            Field(name="session_length", dtype=Int64),
        ],
        source=session_push,
    )
    user_stats = FeatureView(
        name="user_stats",
        entities=[user],
        ttl=timedelta(days=2),
        schema=[Field(name=f, dtype=Float64) for f in USER_FEATURES],
        source=FileSource(name="user_snapshots", path=str(data_dir / "user_features.parquet"), timestamp_field="event_timestamp"),
    )
    return [user, session_push, session, user_stats]


def get_store(cfg: Config) -> FeatureStore:
    os.environ.setdefault("FEAST_REGISTRY", str(cfg.processed_dir / "feast_registry.db"))
    os.environ.setdefault("REDIS_CONNECTION", "localhost:6380")
    return FeatureStore(repo_path=str(REPO))


def apply(store: FeatureStore, data_dir: Path) -> None:
    store.apply(definitions(data_dir))


def historical_features(store: FeatureStore, requests: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time join of session and user features onto (request_id, uid, request_ts)."""
    entity_df = requests.assign(event_timestamp=pd.to_datetime(requests["request_ts"].map(to_datetime), utc=True))
    out = store.get_historical_features(
        entity_df=entity_df[["request_id", "uid", "request_ts", "event_timestamp"]], features=ONLINE_FEATURES
    )
    return out.to_df().sort_values("request_id").reset_index(drop=True)


def online_features(store: FeatureStore, uid: int) -> dict:
    row = store.get_online_features(features=ONLINE_FEATURES, entity_rows=[{"uid": uid}]).to_dict()
    return {k: v[0] for k, v in row.items()}


def push_session_rows(store: FeatureStore, rows: list[dict]) -> None:
    store.push(PUSH_SOURCE, pd.DataFrame(rows), to=PushMode.ONLINE)


def init_online(cfg: Config, data_dir: Path) -> None:
    """Register features and load the latest user snapshot into Redis (the `feast` compose job)."""
    store = get_store(cfg)
    apply(store, data_dir)
    store.materialize(to_datetime(cfg.val_start + 1), to_datetime(cfg.test_start + 1), feature_views=["user_stats"])
    print("feast: applied definitions and materialised user_stats")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["init"])
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config()
    init_online(cfg, args.data_dir or cfg.processed_dir)
