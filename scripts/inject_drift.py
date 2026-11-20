"""End-to-end drift demo against the running stack.

1. send normal traffic and save it as the drift reference window
2. send more normal traffic and check it: no drift expected
3. inject synthetic skip-heavy sessions for 300 users into Redpanda (a broken client
   that reports every play as a skip), send traffic for those users, check again
4. wait for Prometheus to fire the FeatureDrift alert

    python scripts/inject_drift.py
"""

import json
import random
import time

import httpx
import polars as pl
import redis
from confluent_kafka import Producer

from recsys.config import ROOT, load_config
from recsys.drift import drift, publish, served_window

API = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"


def traffic(users: list[int], n: int) -> None:
    with httpx.Client(timeout=5) as client:
        for _ in range(n):
            client.get(f"{API}/recommend", params={"user_id": random.choice(users), "k": 10})


def watermark() -> int:
    r = httpx.get(f"{PROMETHEUS}/api/v1/query", params={"query": "stream_watermark_ts"}).json()
    return int(float(r["data"]["result"][0]["value"][1]))


def check(redis_client, reference, label: str) -> float:
    share, per_column, _ = drift(reference, served_window(redis_client, 1500))
    publish(share, per_column, "localhost:9091")
    drifted = {c: round(s, 2) for c, s in per_column.items() if s > 0.1}
    print(f"{label}: drift share {share:.2f}, drifted features {drifted}")
    return share


def main() -> None:
    random.seed(0)
    cfg = load_config()
    r = redis.Redis(port=6380)
    users = pl.read_parquet(ROOT / "data/processed/test_requests.parquet")["uid"].unique().to_list()
    items = pl.read_parquet(ROOT / "data/processed/replay_events.parquet")["item_id"].unique().to_list()

    print("1. baseline traffic -> reference window")
    traffic(users, 1500)
    reference = served_window(r, 1500)
    reference.to_parquet(cfg.processed_dir / "drift_reference.parquet")

    print("2. more normal traffic")
    traffic(users, 1500)
    check(r, reference, "normal")

    print("3. injecting skip-heavy sessions")
    drifted_users = random.sample(users, 300)
    producer = Producer({"bootstrap.servers": "127.0.0.1:19092"})
    ts = watermark() + 5
    for uid in drifted_users:
        for i in range(40):
            event = {"uid": uid, "timestamp": ts + i * 5, "item_id": random.choice(items), "event_type": "listen",
                     "played_ratio_pct": 3, "is_organic": 0, "emitted_at": time.time()}  # fmt: skip
            producer.produce("events", key=str(uid), value=json.dumps(event))
    producer.flush()
    time.sleep(5)
    traffic(drifted_users, 1500)
    check(r, reference, "after injection")

    print("4. waiting for the alert")
    for _ in range(30):
        alerts = httpx.get(f"{PROMETHEUS}/api/v1/alerts").json()["data"]["alerts"]
        firing = [a for a in alerts if a["labels"]["alertname"] == "FeatureDrift" and a["state"] == "firing"]
        if firing:
            print("FeatureDrift is FIRING:", firing[0]["annotations"]["summary"])
            return
        time.sleep(2)
    print("alert did not fire within 60 s")


if __name__ == "__main__":
    main()
