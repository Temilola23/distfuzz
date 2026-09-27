IMAGE ?= distfuzz:cpu
LIMITS ?= --memory 3g --cpus 4
DOCKER_RUN = docker run --rm $(LIMITS) --shm-size 1g $(IMAGE)

.PHONY: dev lint format typecheck test check docker docker-test docker-shell clean

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

test:
	pytest

check: lint typecheck test

docker:
	docker build -t $(IMAGE) .

docker-test: docker
	$(DOCKER_RUN) python -m pytest -m multirank

docker-shell: docker
	docker run --rm -it $(LIMITS) --shm-size 1g -v $(CURDIR)/runs:/src/runs $(IMAGE) bash

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache src/*.egg-info
