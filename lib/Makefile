# Makefile ──────────────────────────────────────────────────────────────────
.PHONY: dev test build clean publish          # publish is new

VENV       := .venv
TEST_VENV  := /tmp/gai-test
SHELL := /bin/bash

# ── 1. house-keeping ───────────────────────────────────────────────────────
clean:
	rm -rf build/ dist/

# ── 2. editable dev env  (one-off per workstation) ────────────────────────
install: 	
	rm -rf $(VENV)
	uv venv
	source $(VENV)/bin/activate && uv pip install -e ".[dev]"

# ─ 3. bump up the version in pyproject.toml ─────────────────────────────────────────────────
bump:
	source $(VENV)/bin/activate && python -m gai._utils.devtools bump-version $(if $(PART),--part $(PART),)

# ── 4. build artefacts from src/ ───────────────────────────────────────────
build: clean
	source $(VENV)/bin/activate && python -m gai._utils.devtools build

# ── 5. inspect package data ─────────────────────────────────────────────────
inspect:
	source $(VENV)/bin/activate && python -m gai._utils.devtools inspect-pkg-data

# ── 6. smoke-test the built wheel in a fresh venv ─────────────────────────
test:
	source $(VENV)/bin/activate && python test/test_smoke.py

# ── 7. publish to PyPI  (version bump + build + upload) ───────────────────
publish: bump build test
	source $(VENV)/bin/activate && python -m gai._utils.devtools publish

# ── 8. build docker image ───────────────────────────────────────────────────
docker_build:
    # Usage: make docker_build               (use defaults)
    #        make docker_build REPO=myrepo   (custom repo)
    #        make docker_build IMAGE=myimage (custom image)
    #        make docker_build NOCACHE=true  (disable cache)
	source $(VENV)/bin/activate && python -m gai._utils.devtools docker_build \
		$(if $(REPO),--repo-name $(REPO),) \
		$(if $(IMAGE),--image-name $(IMAGE),) \
		$(if $(NOCACHE),--no-cache,)

# ── 9. push docker image to repository ────────────────────────────────────
docker_push:
    # Usage: make docker_push               (use defaults)
    #        make docker_push REPO=myrepo   (custom repo)
    #        make docker_push IMAGE=myimage (custom image)
	source $(VENV)/bin/activate && python -m gai._utils.devtools docker_push \
		$(if $(REPO),--repo-name $(REPO),) \
		$(if $(IMAGE),--image-name $(IMAGE),)

# -------------------------------------------------------------------------
# Usage:
#   make dev          # one-time editable install
#   make test         # rebuild + wheel smoke-test ( /tmp/gai-test venv )
#   make publish      # bump version → build → twine upload
#   make docker_build # build docker image
#   make docker_push   # push docker image to repository
# -------------------------------------------------------------------------