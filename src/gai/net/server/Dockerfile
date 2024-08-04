# Use the official Python runtime as a parent image
FROM python:3.10-slim

# Copy and install gai-sdk
RUN pip install --upgrade "gai-sdk[net]"

# Install any needed packages specified in requirements.txt
WORKDIR /app
COPY src/gai/net ./gai/net
RUN pip install nats-py rich openai
ENV OPENAI_API_KEY=""
WORKDIR /app/gai/net/server
CMD ["python", "main.py"]
