"""Offline pipeline: features -> candidates -> ranker -> evaluation -> MLflow registry.

python -m recsys.train features   # session log, user snapshots, Feast point-in-time joins
python -m recsys.train fit        # bundles, rankers, test evaluation, registration
"""

import argparse
import json
import time
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from recsys import evaluate, registry, store
from recsys.candidates import cooccurrence, popularity
from recsys.config import Config, load_config
from recsys.data import load_artists, load_events
from recsys.features import ITEM_FEATURES, USER_FEATURES, item_snapshot, session_dict, session_feature_log, user_snapshot
from recsys.ranker import ALL_FEATURES, SESSION_RANK_FEATURES, feature_matrix, train_ranker
from recsys.serving import Bundle, build_index, recommend, retrieve, save_bundle, to_index

SOURCES = ["popular", "cooc", "two_tower", "fresh"]


def build_features(cfg: Config) -> None:
    out = cfg.processed_dir
    events = load_events(cfg)
    t = time.time()
    bootstrap_from = cfg.val_start - cfg.history_days * cfg.day
    session_feature_log(
        events.filter(pl.col("timestamp") >= bootstrap_from), cfg.val_start, cfg, out / "session_features.parquet"
    )
    print(f"session log: {time.time() - t:.0f}s")
    pl.concat([user_snapshot(events, cfg.val_start), user_snapshot(events, cfg.test_start)]).write_parquet(
        out / "user_features.parquet"
    )
    fs = store.get_store(cfg)
    store.apply(fs, out)
    for split in ["val", "test"]:
        t = time.time()
        requests = pl.read_parquet(out / f"{split}_requests.parquet").cast({"uid": pl.Int64}).to_pandas()
        store.historical_features(fs, requests).to_parquet(out / f"{split}_features.parquet")
        print(f"{split} point-in-time join: {time.time() - t:.0f}s")


def make_bundle(events, artists, cutoff: int, cfg: Config, index=None) -> tuple[Bundle, np.ndarray]:
    """Bundle as of `cutoff`: learned two-tower parts plus item stats computed before cutoff.

    Returns the bundle and per-item listen counts before cutoff (for the cold-item slice).
    """
    tt = cfg.artifacts_dir / "two_tower"
    item_ids = np.load(tt / "item_ids.npy")
    item_vecs = np.load(tt / "item_vecs.npy")
    snap = pl.DataFrame({"item_id": item_ids}).join(item_snapshot(events, artists, cutoff), on="item_id", how="left")
    n_listens = snap["n_listens"].fill_null(0).to_numpy()
    age = snap["item_age_days"].fill_null(0).to_numpy()  # never played before cutoff -> age 0
    has_vector = np.linalg.norm(item_vecs, axis=1) > 0
    fresh = np.flatnonzero((n_listens < cfg.min_item_count) & (age <= cfg.fresh_days) & has_vector)
    params = {k: getattr(cfg, k) for k in ["hist_len", "cooc_seeds", "cooc_n", "tt_n", "fresh_n", "pool", "mmr_lambda", "nprobe"]}
    bundle = Bundle(
        version=f"cutoff-{cutoff}",
        item_ids=item_ids,
        item_feats=snap.select(ITEM_FEATURES).to_numpy().astype(np.float32),
        artist=snap["artist_id"].fill_null(-1).to_numpy().astype(np.int64),
        item_vecs=item_vecs,
        user_tower=dict(np.load(tt / "user_tower.npz")),
        pop=popularity(events, item_ids, cutoff, cfg),
        fresh=fresh,
        cooc=cooccurrence(events, item_ids, cutoff, cfg),
        index=index if index is not None else build_index(item_vecs, cfg.nprobe, seed=cfg.seed),
        ranker=None,
        features=ALL_FEATURES,
        params=params,
    )
    return bundle, n_listens


def requests_with_positives(bundle: Bundle, split: str, cfg: Config):
    """Yield (request row, session, user, positives, organic flags) with recently played removed."""
    feats = pd.read_parquet(cfg.processed_dir / f"{split}_features.parquet")
    labels = pl.read_parquet(cfg.processed_dir / f"{split}_labels.parquet")
    by_request = {rid: (g["item_id"].to_numpy(), g["is_organic"].to_numpy()) for (rid,), g in labels.group_by("request_id")}
    for row in feats.to_dict("records"):
        if row["request_id"] not in by_request:
            continue
        session = session_dict(row)
        user = {f: row[f] for f in USER_FEATURES}
        item_ids, organic = by_request[row["request_id"]]
        pos = np.searchsorted(bundle.item_ids, item_ids)
        keep = ~np.isin(pos, to_index(bundle, session["last_items"]))
        if keep.any():
            yield row, session, user, pos[keep], organic[keep]


def ranker_dataset(bundle: Bundle, cfg: Config, negatives: int = 200):
    rng = np.random.default_rng(cfg.seed)
    Xs, ys, groups, uids = [], [], [], []
    for row, session, user, pos, _ in requests_with_positives(bundle, "val", cfg):
        cand = retrieve(bundle, session)
        y = np.isin(cand.idx, pos)
        if not y.any():
            continue
        neg = np.flatnonzero(~y)
        # Shuffled: with positives first, tied scores early in boosting would rank them on top
        # by row order alone and early stopping would reward that.
        keep = rng.permutation(np.concatenate([np.flatnonzero(y), rng.choice(neg, min(negatives, len(neg)), replace=False)]))
        Xs.append(feature_matrix(bundle, cand, session, user)[keep])
        ys.append(y[keep].astype(np.int8))
        groups.append(len(keep))
        uids.append(row["uid"])
    return np.concatenate(Xs), np.concatenate(ys), np.asarray(groups), np.asarray(uids)


def split_by_user(X, y, groups, uids, frac: float, seed: int):
    """Hold out a fraction of users (whole query groups) for early stopping."""
    users = np.unique(uids)
    held = set(np.random.default_rng(seed).choice(users, int(len(users) * frac), replace=False))
    is_val = np.array([u in held for u in uids])
    row_val = np.repeat(is_val, groups)
    return (X[~row_val], y[~row_val], groups[~is_val]), (X[row_val], y[row_val], groups[is_val])


def evaluation_records(bundles: dict[str, Bundle], cfg: Config, sasrec: dict | None) -> list[dict]:
    """Run every test request through the serving path of each bundle."""
    main = next(iter(bundles.values()))
    records = []
    for row, session, user, pos, organic in requests_with_positives(main, "test", cfg):
        lists = {}
        for name, b in bundles.items():
            out = recommend(b, session, user, k=10)
            lists[name] = out["ranked"]
            lists[f"{name} + MMR"] = np.searchsorted(b.item_ids, out["items"])
        cand = out["candidates"]
        lists.update({s: cand.sources[s] for s in SOURCES})
        if sasrec is not None:
            lists["sasrec"] = to_index(main, sasrec.get(row["request_id"], []))
        records.append({"uid": row["uid"], "pos": pos, "organic": organic, "union": cand.idx, "lists": lists})
    return records


def figures(records, bundle: Bundle, metrics: dict, rankers: dict, cfg: Config) -> None:
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ks = [10, 25, 50, 100, 150, 200]
    fig, ax = plt.subplots(figsize=(6, 4))
    # the fresh pool is a 30-item list aimed at cold items; it is reported in the slices instead
    for s in ["popular", "cooc", "two_tower"] + (["sasrec"] if "sasrec" in records[0]["lists"] else []):
        ax.plot(ks, [np.mean([evaluate.recall(r["lists"][s], r["pos"], k) for r in records]) for k in ks], marker="o", label=s)
    ax.axhline(metrics["candidates"]["union"]["recall"], color="grey", ls="--", label="union of sources")
    ax.set(xlabel="K", ylabel="Recall@K", title="Candidate recall on the test day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "candidate_recall.png", dpi=150)

    item_pop = np.nan_to_num(bundle.item_feats[:, ITEM_FEATURES.index("item_listens_30d")])
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 5, 26)
    for name, items in [
        ("played (ground truth)", np.concatenate([r["pos"] for r in records])),
        ("popular", np.concatenate([r["lists"]["popular"][:10] for r in records])),
        ("ranker v2 + MMR", np.concatenate([r["lists"]["ranker v2 + MMR"] for r in records])),
    ]:
        ax.hist(np.log10(1 + item_pop[items]), bins=bins, histtype="step", density=True, lw=2, label=name)
    ax.set(xlabel="log10(1 + listens in the last 30 days)", ylabel="density", title="Popularity of recommended vs played items")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "popularity_bias.png", dpi=150)

    booster = rankers["ranker v2"]
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    (gain / gain.sum()).plot.barh(ax=ax)
    ax.set(xlabel="share of total gain", title="Ranker v2 feature importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "ranker_importance.png", dpi=150)
    plt.close("all")


def write_tables(metrics: dict, cfg: Config) -> None:
    def fmt(v):
        return "" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    cols = ["recall@10", "ndcg@10", "distinct", "diversity", "novelty", "pop_percentile"]
    lines = [
        "| method | Recall@10 | NDCG@10 | distinct items | diversity | novelty | pop. percentile |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, r in metrics["methods"].items():
        lines.append(f"| {m} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    lines += ["", "| source | Recall@100 | Recall@200 |", "|---|---|---|"]
    for s, r in metrics["candidates"].items():
        if s == "union":
            lines.append(f"| union (mean {r['mean_size']:.0f} items) | | {r['recall']:.3f} |")
        else:
            lines.append(f"| {s} | {r['recall@100']:.3f} | {r['recall@200']:.3f} |")
    methods = [m for m in next(iter(metrics["slices"].values())) if m != "requests"]
    lines += ["", "| slice | requests | " + " | ".join(methods) + " |", "|---" * (len(methods) + 2) + "|"]
    for s, r in metrics["slices"].items():
        lines.append(f"| {s} | {r['requests']} | " + " | ".join(fmt(r[m]) for m in methods) + " |")
    (cfg.results_dir / "offline_tables.md").write_text("\n".join(lines) + "\n")


def fit(cfg: Config, register: bool = True) -> None:
    t0 = time.time()
    events = load_events(cfg)
    artists = load_artists(cfg)
    val_bundle, _ = make_bundle(events, artists, cfg.val_start, cfg)
    test_bundle, n_listens = make_bundle(events, artists, cfg.test_start, cfg, index=val_bundle.index)
    print(f"bundles built: {time.time() - t0:.0f}s")

    X, y, groups, uids = ranker_dataset(val_bundle, cfg)
    print(f"ranker data: {len(groups)} queries, {len(y)} rows, {y.mean():.3f} positive ({time.time() - t0:.0f}s)")
    tr, va = split_by_user(X, y, groups, uids, 0.1, cfg.seed)
    variants = {
        "ranker v1": [f for f in ALL_FEATURES if f not in SESSION_RANK_FEATURES],
        "ranker v2": ALL_FEATURES,
    }
    rankers = {name: train_ranker(*tr, va, feats, cfg) for name, feats in variants.items()}

    bundles = {}
    for name, booster in rankers.items():
        b = replace(test_bundle, ranker=booster, features=variants[name], version=name)
        save_bundle(b, cfg.artifacts_dir / "bundles" / name.replace(" ", "_"))
        bundles[name] = b

    sasrec_file = cfg.results_dir / "sasrec_test_lists.parquet"
    sasrec = None
    if sasrec_file.exists():
        s = pl.read_parquet(sasrec_file)
        sasrec = dict(zip(s["request_id"].to_list(), s["items"].to_list(), strict=True))

    records = evaluation_records(bundles, cfg, sasrec)
    print(f"evaluated {len(records)} test requests ({time.time() - t0:.0f}s)")
    item_pop = test_bundle.item_feats[:, ITEM_FEATURES.index("item_listens_30d")]
    methods = ["popular", "cooc", "two_tower"] + (["sasrec"] if sasrec else []) + ["ranker v1", "ranker v2", "ranker v2 + MMR"]
    test_feats = pd.read_parquet(cfg.processed_dir / "test_features.parquet")
    user_hist = float(test_feats["user_listens"].fillna(0).quantile(0.2))
    light = set(test_feats.loc[test_feats["user_listens"].fillna(0) <= user_hist, "uid"])
    metrics = {
        "requests": len(records),
        "methods": evaluate.method_table(records, methods, item_pop, test_bundle.item_vecs),
        "v2_minus_v1_recall@10": evaluate.paired_bootstrap(records, "ranker v2", "ranker v1"),
        "mmr_minus_v2_recall@10": evaluate.paired_bootstrap(records, "ranker v2 + MMR", "ranker v2"),
        "candidates": evaluate.candidate_recall(records, SOURCES + (["sasrec"] if sasrec else [])),
        "slices": evaluate.sliced_recall(
            records, ["popular", "cooc", "two_tower", "fresh", "ranker v2 + MMR"], n_listens < cfg.min_item_count, light
        ),
        "light_user_threshold_listens": float(user_hist),
        "rankers": {
            n: {"best_iteration": r.best_iteration, "val_ndcg@10": r.best_score["valid_0"]["ndcg@10"]} for n, r in rankers.items()
        },
    }
    cfg.results_dir.mkdir(exist_ok=True)
    (cfg.results_dir / "offline_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_tables(metrics, cfg)
    figures(records, test_bundle, metrics, rankers, cfg)
    print((cfg.results_dir / "offline_tables.md").read_text())

    if register:
        params = {
            k: getattr(cfg, k)
            for k in ["dim", "hist_len", "num_leaves", "learning_rate", "pop_n", "cooc_n", "tt_n", "mmr_lambda"]
        }
        for name, alias in [("ranker v1", "champion"), ("ranker v2", "challenger")]:
            m = metrics["methods"][name]
            version = registry.register(
                cfg.artifacts_dir / "bundles" / name.replace(" ", "_"),
                name,
                {**params, "features": len(variants[name])},
                {"recall_at_10": m["recall@10"], "ndcg_at_10": m["ndcg@10"], "coverage": m["coverage"]},
                alias,
            )
            print(f"registered {name} as version {version} -> @{alias}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "fit"])
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.stage == "features":
        build_features(cfg)
    else:
        fit(cfg, register=not args.no_register)
