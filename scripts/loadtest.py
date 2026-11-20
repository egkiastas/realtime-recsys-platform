"""Run the k6 load test at several request rates against the compose stack and tabulate results.

k6 runs in a container on the compose network, so it shares the Colima VM's CPUs with the
API; the numbers are a laptop-under-load measurement, not a capacity claim.

    python scripts/loadtest.py                    # 25/50/100/150/200 QPS, 60 s each
    python scripts/loadtest.py --smoke            # CI: 20 QPS for 20 s with pass/fail thresholds
    python scripts/loadtest.py --sequential       # one request at a time: per-stage cost without queueing
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
K6_IMAGE = "grafana/k6:2.4.0"


def wait_for_quiet() -> None:
    """Start each level only when the VM and the Mac are idle; desktop apps share the CPUs."""
    while True:
        vm = subprocess.run(["colima", "ssh", "--", "cat", "/proc/loadavg"], capture_output=True, text=True).stdout
        host = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
        if vm and host and float(vm.split()[0]) < 2 and float(host.split()[1]) < 6:
            return
        time.sleep(10)


def run_k6(qps: int, duration: str, workdir: Path, smoke: bool) -> dict:
    out = workdir / f"summary_{qps}.json"
    cmd = [
        "docker", "run", "--rm", "--network", "recsys_default",
        "-v", f"{workdir}:/out", "-v", f"{ROOT / 'scripts' / 'loadtest.js'}:/scripts/loadtest.js:ro",
        "-e", f"QPS={qps}", "-e", f"DURATION={duration}", "-e", f"OUT=/out/{out.name}",
        "-w", "/out", K6_IMAGE, "run", "--quiet", "/scripts/loadtest.js",
    ]  # fmt: skip
    if smoke:
        cmd[-3:-3] = ["-e", "SMOKE=1"]
    result = subprocess.run(cmd)
    if smoke and result.returncode != 0:
        sys.exit("smoke load test failed its thresholds")
    return json.loads(out.read_text())


def sequential(users: list[int], n: int = 1000) -> dict:
    """One request at a time from the host: per-stage cost without any queueing."""
    import random

    import httpx
    import numpy as np

    random.seed(0)
    stages: dict[str, list[float]] = {}
    with httpx.Client() as c:
        for _ in range(n):
            r = c.get("http://localhost:8000/recommend", params={"user_id": random.choice(users), "k": 10})
            stages.setdefault("client", []).append(r.elapsed.total_seconds() * 1e3)
            for k, v in r.json()["timings_ms"].items():
                stages.setdefault(k, []).append(v)
    return {
        k: {"p50": round(float(np.percentile(v, 50)), 2), "p99": round(float(np.percentile(v, 99)), 2)} for k, v in stages.items()
    }


def row(qps: int, s: dict) -> dict:
    m = s["metrics"]
    reqs = m["http_reqs"]["values"]
    trend = lambda name, p: m[name]["values"][f"p({p})"] if name in m else float("nan")  # noqa: E731
    return {
        "target_qps": qps,
        "achieved_qps": round(reqs["rate"], 1),
        "errors": round(m["http_req_failed"]["values"]["rate"], 4),
        "fallbacks": int(m["fallbacks"]["values"]["count"]) if "fallbacks" in m else 0,
        **{f"client_p{p}": round(trend("http_req_duration", p), 1) for p in (50, 95, 99)},
        **{f"server_p{p}": round(trend("server_total", p), 1) for p in (50, 95, 99)},
        **{f"{stage}_p99": round(trend(f"server_{stage}", 99), 2) for stage in ("features", "retrieve", "rank", "rerank")},
    }


def markdown(rows: list[dict]) -> str:
    head = "| QPS (target / achieved) | client p50 / p95 / p99 ms | server p50 / p95 / p99 ms | p99 by stage: features / retrieve / rank / rerank ms | errors |"
    lines = [head, "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['target_qps']} / {r['achieved_qps']} | {r['client_p50']} / {r['client_p95']} / {r['client_p99']} "
            f"| {r['server_p50']} / {r['server_p95']} / {r['server_p99']} "
            f"| {r['features_p99']} / {r['retrieve_p99']} / {r['rank_p99']} / {r['rerank_p99']} | {r['errors']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qps", type=int, nargs="+", default=[25, 50, 100, 150, 200])
    ap.add_argument("--duration", default="60s")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sequential", action="store_true", help="only the one-request-at-a-time breakdown")
    ap.add_argument("--tag", default="", help="suffix for the results files, e.g. 'no-shadow'")
    args = ap.parse_args()
    if args.smoke:
        args.qps, args.duration = [20], "20s"

    workdir = ROOT / "var" / "loadtest"  # inside $HOME: Colima only shares that with the VM
    workdir.mkdir(parents=True, exist_ok=True)
    users = pl.read_parquet(ROOT / "data/processed/test_requests.parquet")["uid"].unique().to_list()
    (workdir / "users.json").write_text(json.dumps(users))
    if args.sequential:
        result = sequential(users)
        (ROOT / "results" / "latency_sequential.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    run_k6(50, "30s", workdir, smoke=False)  # warm-up, not recorded
    rows = []
    for qps in args.qps:
        if not args.smoke:
            wait_for_quiet()
        rows.append(row(qps, run_k6(qps, args.duration, workdir, args.smoke)))
        print(json.dumps(rows[-1]))
        time.sleep(5)
    if not args.smoke:
        results = ROOT / "results"
        suffix = f"_{args.tag}" if args.tag else ""
        (results / f"loadtest{suffix}.json").write_text(json.dumps(rows, indent=2))
        (results / f"loadtest{suffix}.md").write_text(markdown(rows))
        print(markdown(rows))


if __name__ == "__main__":
    main()
