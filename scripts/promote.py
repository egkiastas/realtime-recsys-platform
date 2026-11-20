"""Move the champion/challenger aliases in the MLflow registry and tell the API to reload.

python scripts/promote.py status
python scripts/promote.py promote    # challenger becomes champion, old champion kept as @previous
python scripts/promote.py rollback   # @previous becomes champion again, demoted version back to @challenger
"""

import argparse
import os
import sys

import httpx
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from recsys.registry import MODEL_NAME, tracking_uri


def alias_version(client: MlflowClient, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias).version
    except MlflowException:
        return None


def status(client: MlflowClient) -> dict:
    return {a: alias_version(client, a) for a in ["champion", "challenger", "previous"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["status", "promote", "rollback"])
    ap.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    args = ap.parse_args()
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    current = status(client)

    if args.action == "promote":
        if not current["challenger"]:
            sys.exit("no challenger to promote")
        client.set_registered_model_alias(MODEL_NAME, "previous", current["champion"])
        client.set_registered_model_alias(MODEL_NAME, "champion", current["challenger"])
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
    elif args.action == "rollback":
        if not current["previous"]:
            sys.exit("nothing to roll back to")
        client.set_registered_model_alias(MODEL_NAME, "challenger", current["champion"])
        client.set_registered_model_alias(MODEL_NAME, "champion", current["previous"])
        client.delete_registered_model_alias(MODEL_NAME, "previous")

    print("aliases:", status(client))
    if args.action != "status":
        try:
            print("api reloaded:", httpx.post(f"{args.api}/admin/reload", timeout=600).json())
        except httpx.HTTPError as e:
            print(f"aliases moved, but the API did not reload ({e}); restart it or POST /admin/reload")


if __name__ == "__main__":
    main()
