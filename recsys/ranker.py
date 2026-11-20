"""Ranker features (computed per request, identically offline and online) and LambdaRank training."""

import lightgbm as lgb
import numpy as np

from recsys.features import ITEM_FEATURES, USER_FEATURES

SESSION_RANK_FEATURES = [
    "listens_30m",
    "skips_30m",
    "likes_30m",
    "skip_rate_30m",
    "organic_share_30m",
    "session_length",
    "n_recent",
]
CROSS_FEATURES = [
    "pop_score",
    "cooc_score",
    "tt_score",
    "recent_sim",
    "artist_recent",
    "same_artist_last",
    "n_sources",
]
ALL_FEATURES = ITEM_FEATURES + USER_FEATURES + SESSION_RANK_FEATURES + CROSS_FEATURES


def feature_matrix(bundle, cand, session: dict, user: dict) -> np.ndarray:
    """One row per candidate, columns in ALL_FEATURES order."""
    idx = cand.idx
    n = len(idx)
    recent = cand.recent_idx
    vecs = bundle.item_vecs[idx]
    tt_score = vecs @ cand.user_vec if cand.user_vec is not None else np.zeros(n, np.float32)
    if len(recent):
        centroid = bundle.item_vecs[recent[:10]].mean(0)
        recent_sim = vecs @ (centroid / (np.linalg.norm(centroid) + 1e-12))
        cand_artist = bundle.artist[idx]
        artists, counts = np.unique(bundle.artist[recent], return_counts=True)
        pos = np.searchsorted(artists, cand_artist).clip(max=len(artists) - 1)
        artist_recent = np.where(artists[pos] == cand_artist, counts[pos], 0)
        same_artist_last = cand_artist == bundle.artist[recent[0]]
    else:
        recent_sim = artist_recent = same_artist_last = np.zeros(n)

    user_row = [np.nan if user.get(f) is None else user[f] for f in USER_FEATURES]
    session_row = [session[f] for f in SESSION_RANK_FEATURES[:-1]] + [len(recent)]
    n_sources = (cand.pop_score > 0).astype(np.int8) + (cand.cooc_score > 0) + cand.in_tt
    return np.column_stack(
        [
            bundle.item_feats[idx],
            np.tile(np.asarray(user_row + session_row, dtype=np.float32), (n, 1)),
            cand.pop_score,
            cand.cooc_score,
            tt_score,
            recent_sim,
            artist_recent,
            same_artist_last,
            n_sources,
        ]
    ).astype(np.float32)


def train_ranker(X: np.ndarray, y: np.ndarray, groups: np.ndarray, valid: tuple, features: list[str], cfg) -> lgb.Booster:
    cols = [ALL_FEATURES.index(f) for f in features]
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": cfg.seed,
        "verbose": -1,
    }
    train = lgb.Dataset(X[:, cols], y, group=groups, feature_name=features)
    Xv, yv, gv = valid
    val = lgb.Dataset(Xv[:, cols], yv, group=gv, reference=train)
    return lgb.train(
        params,
        train,
        num_boost_round=cfg.n_estimators,
        valid_sets=[val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
