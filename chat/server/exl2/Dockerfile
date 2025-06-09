## Stage 1: Build the Exllamav2 wheel

FROM debian:stable-slim AS downloader

RUN --mount=type=cache,target=/root/.cache \
    apt-get update \
    && apt-get install -y wget \
    && apt-get clean
ENV LOADER_FILE="exllamav2-0.1.8+cu121.torch2.2.2-cp310-cp310-linux_x86_64.whl"
RUN wget --no-verbose -O "/tmp/${LOADER_FILE}" \
    "https://github.com/turboderp/exllamav2/releases/download/v0.1.8/${LOADER_FILE}"

# ----------------------------------------------------------------------------------------------------

# Stage 2: Build the final image

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV HOME="/root"
ARG PROJECT_NAME="gai-chat-svr-exl2"
ENV PROJECT_NAME=${PROJECT_NAME}
ENV PROJECT_DIR="/workspaces/${PROJECT_NAME}"
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

ENV LOADER_FILE="exllamav2-0.1.8+cu121.torch2.2.2-cp310-cp310-linux_x86_64.whl"
ARG CATEGORY=ttt
ARG DEVICE=cuda
ENV CATEGORY=${CATEGORY}
ENV MODEL_PATH="${HOME}/.gai/models"

# Step 1: Install system dependencies

# ...

# Install uv

WORKDIR ${PROJECT_DIR}
COPY --from=ghcr.io/astral-sh/uv@sha256:2381d6aa60c326b71fd40023f921a0a3b8f91b14d5db6b90402e65a635053709 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"
RUN echo "Exporting UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
RUN uv venv --system-site-packages
SHELL ["/bin/bash","-c"]
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

# Install exllamav2

COPY --from=downloader "/tmp/${LOADER_FILE}" "/tmp/${LOADER_FILE}"
RUN source ${UV_PROJECT_ENVIRONMENT}/bin/activate \
     && pip install --upgrade pip==24.2 \
     && pip install /tmp/${LOADER_FILE} \
     && rm "/tmp/${LOADER_FILE}"

# Step 2: Copy Source Code

WORKDIR ${PROJECT_DIR}
COPY ./gai-lib          gai-lib
COPY ./src              src
COPY LICENSE            LICENSE
COPY ./pyproject.toml   .

# Step 4: Install project
# (refer to postCreateCommand.sh)

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=$HOME/.cache \    
    source ${UV_PROJECT_ENVIRONMENT}/bin/activate \
    && uv pip install -e .

# Step 5: Startup

RUN echo "{\"app_dir\":\"${HOME}/.gai\"}" > ${HOME}/.gairc
WORKDIR ${PROJECT_DIR}/src/gai/chat/server/api
RUN --mount=type=cache,target=/root/.cache \
    pip install debugpy

ARG BUSTCACHE_STARTUP=0.2
COPY startup.sh .
CMD ["bash","startup.sh"]