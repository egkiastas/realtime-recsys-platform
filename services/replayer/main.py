"""Replays test-day events into Redpanda, `SPEED` dataset seconds per wall-clock second.

Messages are keyed by user so each user's events stay ordered within one partition.
"""

import json
import os
import time

import pyarrow.parquet as pq
from confluent_kafka import Producer
from prometheus_client import Counter, Gauge, start_http_server

SENT = Counter("replayer_events_total", "Events produced")
SIM_TS = Gauge("replayer_sim_ts", "Dataset timestamp the replay has reached")
BEHIND = Gauge("replayer_behind_seconds", "How far the producer lags its wall-clock schedule")


def main() -> None:
    start_http_server(8000)
    speed = float(os.environ.get("SPEED", "60"))
    topic = os.environ.get("EVENTS_TOPIC", "events")
    producer = Producer({"bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"), "linger.ms": 5})
    events = pq.read_table(os.path.join(os.environ.get("DATA_DIR", "/data"), "replay_events.parquet")).to_pylist()
    ts0 = events[0]["timestamp"]
    start = time.time()
    print(f"replaying {len(events)} events at {speed}x", flush=True)
    for i, e in enumerate(events):
        due = start + (e["timestamp"] - ts0) / speed
        wait = due - time.time()
        if wait > 0:
            producer.flush()
            time.sleep(wait)
        else:
            BEHIND.set(-wait)
        e["emitted_at"] = time.time()
        producer.produce(topic, key=str(e["uid"]), value=json.dumps(e))
        producer.poll(0)
        SENT.inc()
        SIM_TS.set(e["timestamp"])
        if i % 20_000 == 0:
            print(f"{i} events, dataset ts {e['timestamp']}", flush=True)
    producer.flush()
    print("replay finished", flush=True)
    while True:  # stay up so Prometheus keeps the final counters
        time.sleep(3600)


if __name__ == "__main__":
    main()
