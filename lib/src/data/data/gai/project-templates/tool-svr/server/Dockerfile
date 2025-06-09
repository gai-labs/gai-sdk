FROM python:3.10-slim-bullseye

ENV HOME="/root"
ARG PROJECT_NAME={{PROJECT_NAME}}
ENV PROJECT_NAME=${PROJECT_NAME}
ENV PROJECT_DIR="/workspaces/${PROJECT_NAME}"
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

# Step 1: Install system dependencies

RUN --mount=type=cache,target=/root/.cache \
    apt update && apt install -y \
    wget \
    && apt autoremove -y \
    && apt clean

# Run as non-root user to avoid file permission issues running certain apps
RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
useradd -l -u ${USER_ID} -g ${USERNAME} -m ${USERNAME} && \
install -d -m 0755 -o ${USERNAME} -g ${USERNAME} /home/${USERNAME}
USER ${USERNAME}
ENV HOME=${HOME}

# Install uv

WORKDIR ${PROJECT_DIR}
COPY --from=ghcr.io/astral-sh/uv@sha256:2381d6aa60c326b71fd40023f921a0a3b8f91b14d5db6b90402e65a635053709 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"
RUN echo "Exporting UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
RUN uv venv --system-site-packages
SHELL ["/bin/bash","-c"]
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

# Step 2: Copy Source Code

WORKDIR ${PROJECT_DIR}
COPY --chown=${USERNAME} ./gai-lib          gai-lib
COPY --chown=${USERNAME} ./src              src
COPY --chown=${USERNAME} LICENSE            LICENSE
COPY --chown=${USERNAME} ./pyproject.toml   .

# Step 5: Startup

RUN echo "{\"app_dir\":\"${HOME}/.gai\"}" > ${HOME}/.gairc
WORKDIR /workspaces/{{PROJECT_NAME}}/src/gai/tools/{{TOOL_NAME}}/svr/api
RUN --mount=type=cache,target=/root/.cache \
    pip install debugpy
COPY startup.sh .
CMD ["bash","startup.sh"]