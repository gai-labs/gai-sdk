# syntax=docker/dockerfile:1.2

FROM torch2.2.0-cuda12.1-ubuntu22.04 as build


# Build Final ----------------------------------------------------------------------------------------

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base
ARG CATEGORY=ttt
ARG DEVICE=cuda
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

# Step 2: Install prebuilt wheels from build

## llama-cpp-python
WORKDIR /app/wheels
COPY --from=build /app/wheels/llama_cpp_python-*.whl /app/wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /app/wheels/llama_cpp_python-*.whl

## exllamav2
WORKDIR /app/wheels
COPY --from=build /app/wheels/exllamav2-*.whl /app/wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /app/wheels/exllamav2-*.whl

# Step 3: Copy Source Code
WORKDIR /app
COPY src/gai/lib src/gai/lib
COPY src/gai/cli src/gai/cli
COPY src/gai/ttt src/gai/ttt
COPY setup.py VERSION .

# Step 4: Build app
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .
RUN pip install debugpy

# Step 5: Startup
WORKDIR /app/src/gai/ttt/server/api
VOLUME /app/gai
RUN echo '{"app_dir":"/app/gai"}' > /root/.gairc
ENV MODEL_PATH="/app/gai/models"
ENV CATEGORY=${CATEGORY}
CMD ["bash","-c","python main.py"]

