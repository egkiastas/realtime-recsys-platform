import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    results_dir: Path

    last_ts: int
    day: int
    gap: int

    requests_per_user: int
    label_window: int
    positive_ratio: int
    seed: int

    last_n: int
    session_window: int
    skip_ratio: int
    history_days: int

    pop_days: int
    pop_n: int
    cooc_window: int
    cooc_topk: int
    cooc_seeds: int
    cooc_n: int
    tt_n: int
    fresh_days: int
    fresh_n: int
    nprobe: int

    dim: int
    hist_len: int
    min_item_count: int
    epochs: int
    batch_size: int
    lr: float
    temperature: float
    id_dropout: float

    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_data_in_leaf: int

    mmr_lambda: float
    pool: int

    feature_timeout_ms: int
    shadow_rate: float

    # Global temporal split: train | gap | val (1 day) | gap | test (1 day)
    @property
    def test_start(self) -> int:
        return self.last_ts - self.day

    @property
    def val_start(self) -> int:
        return self.test_start - self.gap - self.day

    @property
    def val_end(self) -> int:
        return self.test_start - self.gap

    @property
    def train_end(self) -> int:
        return self.val_start - self.gap


def load_config(path: Path | str = ROOT / "config.toml", **overrides) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    flat = {k: v for section in raw.values() for k, v in section.items()}
    flat.update(overrides)
    for f in fields(Config):
        if f.name.endswith("_dir"):
            p = Path(flat[f.name])
            flat[f.name] = p if p.is_absolute() else ROOT / p
    return Config(**flat)
