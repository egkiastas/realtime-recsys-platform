import pytest

from recsys.config import load_config
from recsys.data import load_events, prepare
from tests.fixture import generate


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("recsys")
    cfg = load_config(
        raw_dir=root / "raw", processed_dir=root / "processed", artifacts_dir=root / "artifacts", results_dir=root / "results"
    )
    generate(cfg.raw_dir)
    prepare(cfg)
    return cfg


@pytest.fixture(scope="session")
def events(cfg):
    return load_events(cfg)
