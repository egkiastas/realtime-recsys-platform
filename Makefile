COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo docker-compose) -f infra/docker-compose.yml
RUN := uv run python

.PHONY: data offline up down loadtest drift-demo promote rollback test lint fixture

data:  ## download Yambda-50M (~650 MB on disk) and build the split files
	$(RUN) -m recsys.data download
	$(RUN) -m recsys.data prepare

offline:  ## features, two-tower, rankers, evaluation, registration (needs MLflow up)
	$(COMPOSE) up -d mlflow redis
	$(RUN) -m recsys.train features
	$(RUN) -m recsys.two_tower
	$(RUN) -m recsys.train fit

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

loadtest:
	$(RUN) scripts/loadtest.py

drift-demo:
	$(RUN) scripts/inject_drift.py

promote:
	$(RUN) scripts/promote.py promote

rollback:
	$(RUN) scripts/promote.py rollback

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fixture:  ## tiny synthetic dataset (what CI uses); replaces data/raw
	$(RUN) -m tests.fixture data/raw
	$(RUN) -m recsys.data prepare
