#!/bin/bash
source /${UV_PROJECT_ENVIRONMENT}/bin/activate && \
    python -c "import toml; print(toml.load('/${PROJECT_DIR}/pyproject.toml')['project']['version'])" && \
    uv run main.py