# Use the astral Python 3.13 image.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY requirements.txt .
RUN uv pip install -r requirements.txt  --system

COPY . .
