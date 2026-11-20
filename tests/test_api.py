"""API contract, fallback paths and shadow scoring, with real bundles built from the fixture."""

import time
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from recsys.config import Config
from recsys.data import load_artists
from recsys.features import USER_FEATURES
from recsys.ranker import ALL_FEATURES, SESSION_RANK_FEATURES
from recsys.serving import save_bundle
from recsys.train import make_bundle


def tiny_ranker(features: list[str], seed: int) -> lgb.Booster:
    rng = np.random.default_rng(seed)
    X = rng.random((600, len(features)))
    y = (rng.random(600) < 0.1).astype(int)
    return lgb.train({"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y, group=[60] * 10), 5)


@pytest.fixture(scope="module")
def bundles(cfg: Config, events, tmp_path_factory):
    """Champion and challenger bundles: real candidate data, random tower and ranker weights."""
    rng = np.random.default_rng(0)
    tt = cfg.artifacts_dir / "two_tower"
    tt.mkdir(parents=True, exist_ok=True)
    item_ids = np.sort(events["item_id"].unique().to_numpy())
    vecs = rng.normal(size=(len(item_ids), 16)).astype(np.float32)
    np.save(tt / "item_ids.npy", item_ids)
    np.save(tt / "item_vecs.npy", vecs / np.linalg.norm(vecs, axis=1, keepdims=True))
    np.savez(tt / "user_tower.npz", w1=rng.normal(size=(16, 32)), b1=np.zeros(32), w2=rng.normal(size=(32, 16)), b2=np.zeros(16))

    base, _ = make_bundle(events, load_artists(cfg), cfg.test_start, cfg)
    no_session = [f for f in ALL_FEATURES if f not in SESSION_RANK_FEATURES]
    root = tmp_path_factory.mktemp("bundles")
    paths = {}
    for name, feats, seed in [("champion", no_session, 1), ("challenger", ALL_FEATURES, 2)]:
        save_bundle(replace(base, version=name, features=feats, ranker=tiny_ranker(feats, seed)), root / name)
        paths[name] = root / name
    return base, paths


@pytest.fixture
def client(bundles, monkeypatch):
    base, paths = bundles
    monkeypatch.setenv("CHAMPION_PATH", str(paths["champion"]))
    monkeypatch.setenv("CHALLENGER_PATH", str(paths["challenger"]))
    from services.api import main

    recent = base.item_ids[:5].tolist()
    row = {
        "last_items": recent,
        "last_ts": 1,
        "listens_30m": 5,
        "skips_30m": 1,
        "likes_30m": 0,
        "skip_rate_30m": 0.2,
        "organic_share_30m": 0.4,
        "session_length": 6,
        **{f: 10.0 for f in USER_FEATURES},
    }
    monkeypatch.setattr(main, "fetch_features", lambda uid: dict(row))
    monkeypatch.setattr(main, "SHADOW_RATE", 1.0)
    with TestClient(main.app) as c:
        c.recent = recent
        c.main = main
        yield c


def test_contract(client):
    r = client.get("/recommend", params={"user_id": 42, "k": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 42 and body["fallback"] is None
    assert len(body["items"]) == 7 and len(set(body["items"])) == 7
    assert not set(body["items"]) & set(client.recent), "recently played items must be filtered"
    assert body["model_version"] == "recsys/champion"
    assert {"features", "retrieve", "rank", "rerank", "total"} <= set(body["timings_ms"])
    assert body["timings_ms"]["total"] >= body["timings_ms"]["rank"]


@pytest.mark.parametrize("params", [{"user_id": -1}, {"user_id": "abc"}, {"user_id": 1, "k": 0}, {"user_id": 1, "k": 101}, {}])
def test_rejects_invalid_requests(client, params):
    assert client.get("/recommend", params=params).status_code == 422


def popular(client, k):
    b = client.main.state["champion"]
    return b.item_ids[b.pop[:k]].tolist()


def test_falls_back_when_feature_store_errors(client, monkeypatch):
    def down(uid):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(client.main, "fetch_features", down)
    body = client.get("/recommend", params={"user_id": 1, "k": 5}).json()
    assert body["fallback"] == "feature_error"
    assert body["items"] == popular(client, 5)


def test_falls_back_when_feature_store_is_slow(client, monkeypatch):
    def slow(uid):
        time.sleep(0.3)
        return {}

    monkeypatch.setattr(client.main, "fetch_features", slow)
    t = time.perf_counter()
    body = client.get("/recommend", params={"user_id": 1, "k": 5}).json()
    assert body["fallback"] == "feature_timeout"
    assert time.perf_counter() - t < 0.25, "the timeout must bound the request, not wait for Redis"


def test_falls_back_when_model_fails(client, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("bad model")

    monkeypatch.setattr(client.main, "recommend", broken)
    body = client.get("/recommend", params={"user_id": 1, "k": 5}).json()
    assert body["fallback"] == "model_error"
    assert body["items"] == popular(client, 5)


def overlap_count(main) -> float:
    return next(s.value for m in main.SHADOW_OVERLAP.collect() for s in m.samples if s.name.endswith("_count"))


def test_shadow_scores_challenger_without_changing_response(client):
    before = overlap_count(client.main)
    body = client.get("/recommend", params={"user_id": 3, "k": 10}).json()
    assert body["model_version"] == "recsys/champion"
    assert overlap_count(client.main) == before + 1


def test_shadow_failure_does_not_affect_response(client, monkeypatch):
    challenger = client.main.state["challenger"]
    monkeypatch.setitem(client.main.state, "challenger", replace(challenger, ranker=None))
    before = overlap_count(client.main)
    r = client.get("/recommend", params={"user_id": 3, "k": 10})
    assert r.status_code == 200 and r.json()["fallback"] is None
    assert overlap_count(client.main) == before
