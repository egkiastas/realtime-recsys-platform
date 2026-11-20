import numpy as np
import torch

from recsys.evaluate import ndcg, recall
from recsys.rerank import mmr
from recsys.serving import user_vector
from recsys.two_tower import TwoTower


def test_numpy_user_tower_matches_torch():
    """Serving reimplements the user tower in numpy; it must match the trained module."""
    torch.manual_seed(0)
    n, dim = 50, 8
    model = TwoTower(torch.randn(n, 16), torch.ones(n), torch.arange(n), n, dim, id_dropout=0.0).eval()
    with torch.no_grad():
        item_vecs = model.item_vec(torch.arange(n)).numpy()
        hist = torch.tensor([[3, 7, 7, 12, -1, -1]])
        expected = model.user_vec(hist)[0].numpy()
    mlp = model.user_mlp
    weights = {
        "w1": mlp[0].weight.detach().numpy().T,
        "b1": mlp[0].bias.detach().numpy(),
        "w2": mlp[2].weight.detach().numpy().T,
        "b2": mlp[2].bias.detach().numpy(),
    }
    got = user_vector(item_vecs, weights, np.array([3, 7, 7, 12]))
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert user_vector(item_vecs, weights, np.array([], dtype=int)) is None


def test_cold_item_gets_vector_from_audio_alone():
    """Items without an id embedding (padding index) must still get a non-zero vector."""
    torch.manual_seed(0)
    n, n_ids = 10, 5
    id_map = torch.tensor([0, 1, 2, 3, 4] + [n_ids] * 5)
    model = TwoTower(torch.randn(n, 16), torch.ones(n), id_map, n_ids, 8, id_dropout=0.0).eval()
    with torch.no_grad():
        v = model.item_vec(torch.arange(n))
    assert torch.allclose(v.norm(dim=1), torch.ones(n), atol=1e-5)


def test_mmr_without_diversity_is_plain_ranking():
    rng = np.random.default_rng(0)
    rel = rng.random(30)
    vecs = rng.normal(size=(30, 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    assert list(mmr(rel, vecs, 10, lam=1.0)) == list(np.argsort(-rel)[:10])


def test_mmr_pushes_down_near_duplicates():
    base = np.eye(4)
    vecs = np.vstack([base[0], base[0], base[1], base[2]])  # items 0 and 1 are identical
    rel = np.array([1.0, 0.95, 0.6, 0.5])
    assert list(mmr(rel, vecs, 3, lam=1.0)) == [0, 1, 2]
    assert 1 not in mmr(rel, vecs, 3, lam=0.5)
    assert len(mmr(rel, vecs, 10, lam=0.5)) == 4


def test_metrics():
    pos = np.array([5, 9])
    assert recall(np.array([5, 1, 9]), pos, 10) == 1.0
    assert recall(np.array([1, 2, 5]), pos, 2) == 0.0
    assert ndcg(np.array([5, 9, 1]), pos, 10) == 1.0
    assert 0 < ndcg(np.array([1, 5, 9]), pos, 10) < 1.0
