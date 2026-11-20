"""Feature drift job: compare recently served features against a reference window.

The API logs the features of every non-fallback request to a capped Redis list. This job
reads the latest window, runs Evidently's drift preset against a saved reference window,
and pushes the drifted-column share to the Prometheus Pushgateway, where an alert rule
(infra/prometheus/alerts.yml) fires above 30%. In production it would run on a schedule.

    python -m recsys.drift --save-reference   # snapshot a healthy window
    python -m recsys.drift                    # check the latest window
"""

import argparse
import json
import os

import pandas as pd
import redis
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from recsys.config import load_config

LOG_KEY = "served_features"


def served_window(client: redis.Redis, n: int) -> pd.DataFrame:
    rows = [json.loads(r) for r in client.lrange(LOG_KEY, 0, n - 1)]
    return pd.DataFrame(rows).astype(float)


def drift(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[float, dict, object]:
    cols = list(reference.columns)
    definition = DataDefinition(numerical_columns=cols)
    snapshot = Report([DataDriftPreset(num_method="wasserstein", threshold=0.1)]).run(
        Dataset.from_pandas(current[cols], data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    share, per_column = 0.0, {}
    for m in snapshot.dict()["metrics"]:
        if m["metric_name"].startswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])
        elif m["metric_name"].startswith("ValueDrift"):
            per_column[m["config"]["column"]] = float(m["value"])
    return share, per_column, snapshot


def publish(share: float, per_column: dict, gateway: str) -> None:
    reg = CollectorRegistry()
    Gauge("recsys_feature_drift_share", "Share of served features that drifted", registry=reg).set(share)
    g = Gauge("recsys_feature_drift_score", "Wasserstein distance (normed) per feature", ["feature"], registry=reg)
    for col, score in per_column.items():
        g.labels(col).set(score)
    push_to_gateway(gateway, job="drift", registry=reg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-reference", action="store_true")
    ap.add_argument("--window", type=int, default=2000, help="most recent served requests to compare")
    args = ap.parse_args()
    cfg = load_config()
    client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380"))
    ref_path = cfg.processed_dir / "drift_reference.parquet"
    current = served_window(client, args.window)
    if args.save_reference:
        current.to_parquet(ref_path)
        print(f"saved reference window of {len(current)} requests")
        return
    share, per_column, snapshot = drift(pd.read_parquet(ref_path), current)
    report = cfg.artifacts_dir / "drift_report.html"
    report.parent.mkdir(exist_ok=True)
    snapshot.save_html(str(report))
    publish(share, per_column, os.environ.get("PUSHGATEWAY", "localhost:9091"))
    drifted = {c: round(s, 3) for c, s in per_column.items() if s > 0.1}
    print(json.dumps({"window": len(current), "drift_share": round(share, 3), "drifted": drifted}))


if __name__ == "__main__":
    main()
