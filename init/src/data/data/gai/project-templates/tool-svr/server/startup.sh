#!/usr/bin/env bash
set -e

# 2) activate the virtualenv
source "/workspaces/${PROJECT_NAME}/.venv/bin/activate"

# 4) print your project version
uv add toml
python -c "import toml; print(\"version:\"+toml.load('/workspaces/${PROJECT_NAME}/pyproject.toml')['project']['version'])"

# 5) start api
python main.py
