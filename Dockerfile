# Start from a Python 3.10 base image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies (CPU versions)
RUN pip install -r requirements.txt

# Copy your Python files into the container
COPY . .

# Add these debug lines
RUN python -c "import sys; print(sys.path)"
RUN ls -la /app
RUN pip list