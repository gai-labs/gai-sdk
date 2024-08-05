# Use the official Python runtime as a parent image
FROM python:3.10-slim

# Define ARG for GAI_SDK_VERSION
ARG GAI_SDK_VERSION=""

# Copy and conditionally install gai-sdk based on GAI_SDK_VERSION
RUN if [ -z "$GAI_SDK_VERSION" ]; then pip install --upgrade "gai-sdk[net]"; else pip install "gai-sdk[net]==$GAI_SDK_VERSION"; fi

# Startup
RUN echo '{"app_dir":"/app/.gai"}' > /root/.gairc
VOLUME /app/.gai

# Install any needed packages specified in requirements.txt
WORKDIR /app
COPY src/gai/net/server ./src/gai/net/server
ENV OPENAI_API_KEY=""
WORKDIR /app/src/gai/net/server
CMD ["python", "main.py"]
