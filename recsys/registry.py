"""MLflow registry: one registered model ("recsys") whose versions are full serving bundles.

A bundle holds everything the request path needs (item vectors, Faiss index, co-occurrence
graph, popularity list, item features, ranker), so a version is deployable on its own and
promotion/rollback is a single alias move.
"""

import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

MODEL_NAME = "recsys"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5002")


class BundleModel(mlflow.pyfunc.PythonModel):
    """Lets the bundle be scored through mlflow pyfunc too; the API loads the files directly."""

    def load_context(self, context):
        from recsys.serving import load_bundle

        self.bundle = load_bundle(Path(context.artifacts["bundle"]))

    def predict(self, context, model_input, params=None):
        from recsys.serving import recommend

        k = (params or {}).get("k", 10)
        return [
            recommend(self.bundle, s, u, k)["items"] for s, u in zip(model_input["session"], model_input["user"], strict=True)
        ]


def register(bundle_dir: Path, run_name: str, params: dict, metrics: dict, alias: str | None = None) -> str:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("recsys")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        info = mlflow.pyfunc.log_model(
            name="bundle",
            python_model=BundleModel(),
            artifacts={"bundle": str(bundle_dir)},
            registered_model_name=MODEL_NAME,
        )
    version = str(info.registered_model_version)
    if alias:
        MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)
    return version


def download(alias: str, cache_dir: Path) -> tuple[Path, str]:
    """Fetch the bundle behind an alias into cache_dir/<version>; returns (bundle_dir, version).

    Versions are immutable, so a version already on disk (e.g. the one being rolled back to)
    is not downloaded again.
    """
    mlflow.set_tracking_uri(tracking_uri())
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
    dst = cache_dir / mv.version
    if not dst.exists():
        tmp = Path(tempfile.mkdtemp(dir=cache_dir))
        mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/{mv.version}", dst_path=str(tmp))
        try:
            tmp.rename(dst)
        except OSError:  # another API worker finished the same download first
            shutil.rmtree(tmp)
    # pyfunc copies the bundle directory under its own name inside artifacts/
    (bundle_dir,) = (dst / "artifacts").iterdir()
    return bundle_dir, mv.version
