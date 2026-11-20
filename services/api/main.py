"""Recommendation API: Feast features -> retrieve -> rank -> re-rank, with shadow scoring.

Endpoints are sync on purpose: the work is CPU-bound numpy/LightGBM, so it runs in
Starlette's thread pool instead of blocking the event loop.
"""

import contextlib
import gc
import json
import logging
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import redis
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel

from recsys import registry, store
from recsys.config import load_config
from recsys.features import USER_FEATURES, session_dict
from recsys.serving import load_bundle, popular_fallback, recommend


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": round(record.created, 3), "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
log = logging.getLogger("api")
log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False

LATENCY_BUCKETS = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5)
STAGE_SECONDS = Histogram("recsys_stage_seconds", "Per-stage latency", ["stage"], buckets=LATENCY_BUCKETS)
REQUEST_SECONDS = Histogram("recsys_request_seconds", "End-to-end latency inside the handler", buckets=LATENCY_BUCKETS)
REQUESTS = Counter("recsys_requests_total", "Recommendation requests", ["outcome"])
FALLBACKS = Counter("recsys_fallback_total", "Requests served from the popularity fallback", ["reason"])
SHADOW_OVERLAP = Histogram(
    "recsys_shadow_overlap", "overlap@k between champion and challenger", buckets=[i / 10 for i in range(11)]
)
MODEL_INFO = Gauge("recsys_model_info", "Loaded model versions", ["alias", "version"])

# Served features are logged per request for the drift job; the list is capped in Redis.
DRIFT_SESSION_FEATURES = ["listens_30m", "skips_30m", "skip_rate_30m", "organic_share_30m", "session_length"]
SERVED_LOG_KEY, SERVED_LOG_MAX = "served_features", 20_000

cfg = load_config()
SHADOW_RATE = float(os.environ.get("SHADOW_RATE") or cfg.shadow_rate)
BUNDLE_CACHE = Path(os.environ.get("BUNDLE_CACHE", tempfile.gettempdir())) / "recsys-bundles"
BUNDLE_CACHE.mkdir(parents=True, exist_ok=True)
state: dict = {}
feature_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="features")


def load_models() -> dict:
    """Bundles from local dirs (tests/CI) or from the MLflow aliases. No challenger is fine."""
    models = {}
    for alias in ["champion", "challenger"]:
        local = os.environ.get(f"{alias.upper()}_PATH")
        if local:
            models[alias] = load_bundle(Path(local))
            continue
        try:
            path, version = registry.download(alias, BUNDLE_CACHE)
        except Exception as e:
            if alias == "champion":
                raise
            log.info("no challenger loaded", extra={"fields": {"error": str(e)[:200]}})
            continue
        models[alias] = load_bundle(path, version=f"v{version}")
    return models


def warm_up(models: dict) -> None:
    """The first Feast read loads the registry, and the memory-mapped bundle arrays are paged
    in lazily (50-200 ms on first touch); do both before serving, not on users' requests."""
    try:
        fetch_features(0)
    except Exception as e:
        log.warning("feature store not reachable during warm-up", extra={"fields": {"error": repr(e)[:200]}})
    for bundle in models.values():
        for arr in (bundle.item_vecs, bundle.item_feats, bundle.artist, *bundle.cooc):
            np.asarray(arr).sum()  # fault the memory-mapped pages in once
        session = {"last_items": bundle.item_ids[bundle.pop[:20]].tolist(), "listens_30m": 0, "skips_30m": 0,
                   "likes_30m": 0, "skip_rate_30m": 0.0, "organic_share_30m": 0.0, "session_length": 0}  # fmt: skip
        for _ in range(3):
            recommend(bundle, session, {}, 10)


def activate(models: dict) -> None:
    """Swap in fully loaded and warmed models in one step; requests never see a half-reload."""
    warm_up(models)
    state["champion"] = models["champion"]
    if "challenger" in models:
        state["challenger"] = models["challenger"]
    else:
        state.pop("challenger", None)
    MODEL_INFO.clear()
    for alias, bundle in models.items():
        MODEL_INFO.labels(alias, bundle.version).set(1)
        log.info("model active", extra={"fields": {"alias": alias, "version": bundle.version}})


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["store"] = store.get_store(cfg)
    host, port = os.environ.get("REDIS_CONNECTION", "localhost:6380").split(":")
    state["redis"] = redis.Redis(host=host, port=int(port), socket_timeout=0.05)  # served-feature log
    activate(load_models())
    # Everything loaded so far lives for the whole process; keep the collector from rescanning it.
    gc.freeze()
    yield


app = FastAPI(title="recsys", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class Recommendation(BaseModel):
    user_id: int
    items: list[int]
    model_version: str
    fallback: str | None
    timings_ms: dict[str, float]


def fetch_features(uid: int) -> dict:
    return store.online_features(state["store"], uid)


def shadow_score(session: dict, user: dict, k: int, champion_items: list[int], uid: int) -> None:
    challenger = state.get("challenger")
    if challenger is None:
        return
    try:
        items = recommend(challenger, session, user, k)["items"]
        overlap = len(set(items) & set(champion_items)) / k
        SHADOW_OVERLAP.observe(overlap)
        log.info("shadow", extra={"fields": {"user_id": uid, "challenger": challenger.version, "overlap_at_k": overlap}})
    except Exception as e:
        log.warning("shadow failed", extra={"fields": {"error": repr(e)}})


def log_served(session: dict, user: dict) -> None:
    row = {f: session[f] for f in DRIFT_SESSION_FEATURES}
    row["n_recent"] = len(session["last_items"])
    row.update({f: user.get(f) for f in USER_FEATURES})
    with contextlib.suppress(redis.RedisError):  # monitoring must never fail a request
        state["redis"].pipeline().lpush(SERVED_LOG_KEY, json.dumps(row)).ltrim(SERVED_LOG_KEY, 0, SERVED_LOG_MAX - 1).execute()


@app.get("/recommend", response_model=Recommendation)
def recommend_endpoint(
    background: BackgroundTasks,
    user_id: int = Query(ge=0, le=2**32 - 1),
    k: int = Query(10, ge=1, le=100),
):
    t0 = time.perf_counter()
    champion = state["champion"]
    timings: dict[str, float] = {}
    fallback = None
    try:
        row = feature_pool.submit(fetch_features, user_id).result(timeout=cfg.feature_timeout_ms / 1000)
        session = session_dict(row)
        user = {f: row.get(f) for f in USER_FEATURES}
    except FutureTimeout:
        fallback = "feature_timeout"
    except Exception as e:
        fallback = "feature_error"
        log.warning("feature fetch failed", extra={"fields": {"user_id": user_id, "error": repr(e)[:200]}})
    timings["features"] = (time.perf_counter() - t0) * 1e3

    if fallback is None:
        try:
            out = recommend(champion, session, user, k)
            items = out["items"]
            timings.update(out["timings"])
        except Exception as e:
            fallback = "model_error"
            log.error("pipeline failed", extra={"fields": {"user_id": user_id, "error": repr(e)[:200]}})
    if fallback is not None:
        items = popular_fallback(champion, k)
        FALLBACKS.labels(fallback).inc()
    else:
        background.add_task(log_served, session, user)
        if state.get("challenger") is not None and random.random() < SHADOW_RATE:
            background.add_task(shadow_score, session, user, k, items, user_id)

    timings["total"] = (time.perf_counter() - t0) * 1e3
    for stage, ms in timings.items():
        if stage != "total":
            STAGE_SECONDS.labels(stage).observe(ms / 1e3)
    REQUEST_SECONDS.observe(timings["total"] / 1e3)
    REQUESTS.labels("fallback" if fallback else "ok").inc()
    log.info(
        "recommend",
        extra={
            "fields": {
                "user_id": user_id,
                "k": k,
                "version": champion.version,
                "fallback": fallback,
                **{f"{s}_ms": round(v, 2) for s, v in timings.items()},
            }
        },
    )
    return Recommendation(
        user_id=user_id,
        items=items,
        model_version=f"recsys/{champion.version}",
        fallback=fallback,
        timings_ms={s: round(v, 3) for s, v in timings.items()},
    )


@app.post("/admin/reload")
def reload_models():
    """Re-resolve the MLflow aliases; called by scripts/promote.py after moving an alias."""
    try:
        activate(load_models())
    except Exception as e:
        raise HTTPException(500, f"reload failed: {e}") from e
    return {alias: state[alias].version for alias in ["champion", "challenger"] if alias in state}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "champion": state["champion"].version,
        "challenger": getattr(state.get("challenger"), "version", None),
    }
