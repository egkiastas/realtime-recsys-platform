"""Two-tower retrieval with audio embeddings, trained with in-batch softmax + logQ correction.

Item tower: MLP(audio embedding) + ID embedding. IDs exist only for items with enough
training listens and are randomly dropped during training, so the audio path alone
produces usable vectors for cold-start items.
User tower: mean of the item vectors of the last `hist_len` listens plus a residual MLP.
The serving side only needs the exported item vectors and the small MLP (serving.user_vector).
"""

import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from recsys.config import Config, load_config
from recsys.data import load_events


class TwoTower(nn.Module):
    def __init__(
        self, audio: torch.Tensor, has_audio: torch.Tensor, id_map: torch.Tensor, n_ids: int, dim: int, id_dropout: float
    ):
        super().__init__()
        self.register_buffer("audio", audio)
        self.register_buffer("has_audio", has_audio)
        self.register_buffer("id_map", id_map)
        self.n_ids = n_ids
        self.id_dropout = id_dropout
        self.id_emb = nn.Embedding(n_ids + 1, dim, padding_idx=n_ids)
        nn.init.normal_(self.id_emb.weight, std=0.05)
        with torch.no_grad():
            self.id_emb.weight[n_ids].zero_()
        self.audio_mlp = nn.Sequential(nn.Linear(audio.shape[1], 256), nn.ReLU(), nn.Linear(256, dim))
        self.user_mlp = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, dim))

    def item_vec(self, idx: torch.Tensor) -> torch.Tensor:
        a = self.audio_mlp(self.audio[idx].float()) * self.has_audio[idx].unsqueeze(-1)
        ids = self.id_map[idx]
        if self.training and self.id_dropout > 0:
            ids = torch.where(torch.rand(ids.shape, device=ids.device) < self.id_dropout, self.n_ids, ids)
        return F.normalize(a + self.id_emb(ids), dim=-1)

    def user_vec(self, hist: torch.Tensor) -> torch.Tensor:
        mask = (hist >= 0).float()
        iv = self.item_vec(hist.clamp(min=0)) * mask.unsqueeze(-1)
        m = iv.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(m + self.user_mlp(m), dim=-1)


def training_data(events: pl.DataFrame, item_ids: np.ndarray, cfg: Config, days: int = 120, max_targets: int = 4_000_000):
    """Listen sequences sorted by user/time, user start offsets, and sampled target positions.

    Targets are >=50%-played listens from the last `days` of training whose item is not
    already among the user's previous `hist_len` listens, the same rule that removes
    recently played tracks from labels and recommendations.
    """
    listens = (
        events.filter((pl.col("event_type") == "listen") & (pl.col("timestamp") < cfg.train_end))
        .select("uid", "timestamp", "item_id", "played_ratio_pct")
        .sort("uid", "timestamp", maintain_order=True)
    )
    seq = np.searchsorted(item_ids, listens["item_id"].to_numpy()).astype(np.int64)
    uid = listens["uid"].to_numpy()
    user_start = np.zeros(len(uid), dtype=np.int64)
    boundaries = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
    user_start[boundaries] = boundaries
    user_start = np.maximum.accumulate(user_start)

    ts = listens["timestamp"].to_numpy()
    ratio = listens["played_ratio_pct"].to_numpy()
    cand = np.flatnonzero(
        (ts >= cfg.train_end - days * cfg.day) & (ratio >= cfg.positive_ratio) & (np.arange(len(seq)) > user_start)
    )
    keep = []
    for chunk in np.array_split(cand, max(1, len(cand) // 500_000)):
        hist = gather_history(seq, user_start, chunk, cfg.hist_len)
        keep.append(chunk[~(hist == seq[chunk, None]).any(1)])
    targets = np.concatenate(keep)
    rng = np.random.default_rng(cfg.seed)
    if len(targets) > max_targets:
        targets = rng.choice(targets, max_targets, replace=False)
    return seq, user_start, targets


def gather_history(seq: np.ndarray, user_start: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    """Previous `n` items for each position, most recent first, -1 padded."""
    idx = pos[:, None] - 1 - np.arange(n)[None, :]
    valid = idx >= user_start[pos, None]
    return np.where(valid, seq[np.clip(idx, 0, None)], -1)


def train(cfg: Config) -> None:
    out = cfg.artifacts_dir / "two_tower"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    events = load_events(cfg)
    item_ids = np.sort(events["item_id"].unique().to_numpy())

    emb = np.load(cfg.raw_dir / "embeddings.npz")
    known = np.isin(emb["item_id"], item_ids)  # the embeddings file may cover items without events
    pos = np.searchsorted(item_ids, emb["item_id"][known])
    audio = np.zeros((len(item_ids), emb["embed"].shape[1]), dtype=np.float16)
    audio[pos] = emb["embed"][known]
    has_audio = np.zeros(len(item_ids), dtype=np.float32)
    has_audio[pos] = 1

    seq, user_start, targets = training_data(events, item_ids, cfg)
    counts = np.bincount(seq, minlength=len(item_ids))
    id_items = np.flatnonzero(counts >= cfg.min_item_count)
    id_map = np.full(len(item_ids), len(id_items), dtype=np.int64)
    id_map[id_items] = np.arange(len(id_items))
    target_freq = np.bincount(seq[targets], minlength=len(item_ids)) / len(targets)
    print(f"{len(targets)} targets, {len(id_items)} items with an id embedding, {int(has_audio.sum())} with audio")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TwoTower(
        torch.from_numpy(audio), torch.from_numpy(has_audio), torch.from_numpy(id_map), len(id_items), cfg.dim, cfg.id_dropout
    ).to(device)
    log_q = torch.from_numpy(np.log(target_freq + 1e-12).astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)

    model.train()
    for epoch in range(cfg.epochs):
        t, losses = time.time(), []
        for batch in np.array_split(rng.permutation(targets), max(1, len(targets) // cfg.batch_size)):
            hist = torch.from_numpy(gather_history(seq, user_start, batch, cfg.hist_len)).to(device)
            tgt = torch.from_numpy(seq[batch]).to(device)
            u, v = model.user_vec(hist), model.item_vec(tgt)
            logits = u @ v.T / cfg.temperature - log_q[tgt][None, :]
            same = tgt[:, None] == tgt[None, :]
            logits = logits.masked_fill(same & ~torch.eye(len(tgt), dtype=torch.bool, device=device), -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(tgt), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch}: loss {np.mean(losses):.3f} ({time.time() - t:.0f}s)")

    model.eval()
    with torch.no_grad():
        vecs = [
            model.item_vec(torch.arange(s, min(s + 65_536, len(item_ids)), device=device)).cpu().numpy()
            for s in range(0, len(item_ids), 65_536)
        ]
    np.save(out / "item_vecs.npy", np.concatenate(vecs).astype(np.float32))
    np.save(out / "item_ids.npy", item_ids)
    layers = model.user_mlp
    np.savez(
        out / "user_tower.npz",
        w1=layers[0].weight.detach().cpu().numpy().T,
        b1=layers[0].bias.detach().cpu().numpy(),
        w2=layers[2].weight.detach().cpu().numpy().T,
        b2=layers[2].bias.detach().cpu().numpy(),
    )
    (out / "meta.json").write_text(
        json.dumps({"targets": int(len(targets)), "id_items": int(len(id_items)), "device": str(device)})
    )


if __name__ == "__main__":
    train(load_config())
