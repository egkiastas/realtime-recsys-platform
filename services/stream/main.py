"""Stream processor: Redpanda events -> session features (recsys.features.update) -> Feast push.

On start it rebuilds per-user state from the bootstrap window (the same window the offline
feature log starts from), then reads the events topic from the beginning. Re-reading old
events after a restart is harmless: Feast's Redis store ignores rows older than what it has.
"""

import json
import logging
import os
import time
import uuid

import polars as pl
from confluent_kafka import Consumer
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from recsys import store
from recsys.config import load_config
from recsys.features import SessionState, replay, update

logging.basicConfig(level=logging.INFO, format='{"ts": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}')
log = logging.getLogger("stream")

EVENTS = Counter("stream_events_total", "Events processed")
PUSH_SECONDS = Histogram(
    "stream_push_seconds", "Feast push latency per micro-batch", buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1)
)
LAG = Histogram(
    "stream_event_lag_seconds",
    "Wall time from the replayer emitting an event to its features being in Redis",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
WATERMARK = Gauge("stream_watermark_ts", "Largest event timestamp (dataset seconds) pushed so far")

cfg = load_config()


def bootstrap(fs, events: pl.DataFrame) -> dict:
    """Rebuild per-user state from history and push each user's latest row."""
    states: dict = {}
    last = {}
    for uid, row in replay(events, cfg, states):
        last[uid] = row
    rows = [{"uid": uid, **row} for uid, row in last.items()]
    for i in range(0, len(rows), 2000):
        store.push_session_rows(fs, rows[i : i + 2000])
    return states


def process(fs, states: dict, events: list[dict]) -> list[dict]:
    """Apply a micro-batch of events and push only the latest row per user."""
    latest: dict[int, dict] = {}
    for e in events:
        st = states.get(e["uid"])
        if st is None:
            st = states[e["uid"]] = SessionState()
        row = update(st, e["timestamp"], e["item_id"], e["event_type"], e.get("played_ratio_pct"), e["is_organic"], cfg)
        latest[e["uid"]] = {"uid": e["uid"], **row}
    rows = list(latest.values())
    if rows:
        store.push_session_rows(fs, rows)
    return rows


def main() -> None:
    fs = store.get_store(cfg)
    t = time.time()
    history = pl.read_parquet(os.path.join(os.environ.get("DATA_DIR", "/data"), "bootstrap_events.parquet"))
    states = bootstrap(fs, history)
    log.info(f"bootstrapped {len(states)} users from {len(history)} events in {time.time() - t:.0f}s")
    start_http_server(8000)  # after bootstrap, so the compose healthcheck means "ready"
    consumer = Consumer(
        {
            "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"),
            # a fresh group each start so the whole topic is re-read on top of the bootstrap state
            "group.id": f"stream-{uuid.uuid4().hex[:8]}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([os.environ.get("EVENTS_TOPIC", "events")])
    log.info("consuming")
    while True:
        msgs = consumer.consume(num_messages=500, timeout=0.05)
        events = []
        for m in msgs:
            if m.error():
                log.warning(str(m.error()))
            else:
                events.append(json.loads(m.value()))
        if not events:
            continue
        t = time.perf_counter()
        rows = process(fs, states, events)
        PUSH_SECONDS.observe(time.perf_counter() - t)
        now = time.time()
        for e in events:
            if "emitted_at" in e:
                LAG.observe(now - e["emitted_at"])
        EVENTS.inc(len(events))
        WATERMARK.set(max(r["last_ts"] for r in rows))


if __name__ == "__main__":
    main()
