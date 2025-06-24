VENV       := .venv
TEST_VENV  := /tmp/gai-test
SHELL := /bin/bash

# Makefile ──────────────────────────────────────────────────────────────────
.PHONY: clean

# ── 1. house-keeping ───────────────────────────────────────────────────────
clean:
	rm -rf build/ dist/
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name 'tmp' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.venv' -exec rm -rf {} +
	find . -type d -name 'node_modules' -exec rm -rf {} +

# ── 2. editable dev env  (one-off per workstation) ────────────────────────
install: 
	rm -rf $(VENV)
	uv venv
	source $(VENV)/bin/activate && uv pip install -e ".[dev]"

# ─ 3. bump up the version in pyproject.toml ─────────────────────────────────────────────────
# Usage: make bump            (patch++)
#        make bump PART=minor (minor++)
#        make bump PART=major (major++)
bump:
	source $(VENV)/bin/activate && python -m gai._utils.devtools bump-version $(if $(PART),--part $(PART),)

# ── 4. build artefacts from src/ ───────────────────────────────────────────
build: 
	rm -rf build/ dist/
	source $(VENV)/bin/activate && python -m gai._utils.devtools build

# ── 5. inspect package data ─────────────────────────────────────────────────
inspect:
	source $(VENV)/bin/activate && python -m gai._utils.devtools inspect-pkg-data

# ── 6. smoke-test the built wheel in a fresh venv ─────────────────────────
test:
	source $(VENV)/bin/activate && python -m gai._utils.devtools smoke-test

# ── 7. publish to PyPI  (version bump + build + upload) ───────────────────
publish: bump build test
	source $(VENV)/bin/activate && python -m gai._utils.devtools publish

# ── 8. build docker image ───────────────────────────────────────────────────
docker_build:
    # Usage: make docker_build               (use defaults)
    #        make docker_build REPO=myrepo   (custom repo)
    #        make docker_build IMAGE=myimage (custom image)
    #        make docker_build NOCACHE=true  (disable cache)
	python -m gai._utils.devtools docker_build \
		$(if $(REPO),--repo-name $(REPO),) \
		$(if $(IMAGE),--image-name $(IMAGE),) \
		$(if $(NOCACHE),--no-cache,)

# ── 9. push docker image to repository ────────────────────────────────────
docker_push:
    # Usage: make docker_push               (use defaults)
    #        make docker_push REPO=myrepo   (custom repo)
    #        make docker_push IMAGE=myimage (custom image)
	python -m gai._utils.devtools docker_push \
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