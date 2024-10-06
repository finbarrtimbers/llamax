# Start from a Python 3.10 base image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU versions)
RUN pip install --no-cache-dir \
    jax \
    jaxlib \
    flax \
    torch \
    numpy \
    fairscale \
    pytest

# Copy your Python files into the container
COPY . .

WORKDIR llamax