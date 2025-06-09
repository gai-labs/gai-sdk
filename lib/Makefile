# Makefile ──────────────────────────────────────────────────────────────────
.PHONY: dev test build clean publish          # publish is new

VENV       := $(HOME)/.venv
TEST_VENV  := /tmp/gai-test

# ── 1. house-keeping ───────────────────────────────────────────────────────
clean:
	rm -rf build/ dist/

# ── 2. editable dev env  (one-off per workstation) ────────────────────────
dev: $(VENV)/bin/activate
$(VENV)/bin/activate:
	python -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev]"

# ─ 3. bump up the version in pyproject.toml ─────────────────────────────────────────────────
bump:
	# Usage: make bump            (patch++)
	#        make bump PART=minor (minor++)
	#        make bump PART=major (major++)
	python -m gai._utils.devtools bump-version $(if $(PART),--part $(PART),)

# ── 4. build artefacts from src/ ───────────────────────────────────────────
build: clean
	python -m gai._utils.devtools build

# ── 5. inspect package data ─────────────────────────────────────────────────
inspect:
	python -m gai._utils.devtools inspect-pkg-data

# ── 6. smoke-test the built wheel in a fresh venv ─────────────────────────
test: build
	python -m gai._utils.devtools testdist

	#rm -rf $(TEST_VENV)
	#python -m venv $(TEST_VENV)
	#$(TEST_VENV)/bin/pip install dist/*.whl
	#$(TEST_VENV)/bin/gai init --force   # verifies resources copy

# ── 7. publish to PyPI  (version bump + build + upload) ───────────────────
publish:
	python -m gai._utils.devtools publish

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