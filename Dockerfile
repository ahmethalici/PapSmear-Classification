# Use a specific, slim, and secure base image
FROM python:3.9-slim-bookworm

LABEL maintainer="Ahmet Halici <dev.ahmet.halici@gmail.com>"
LABEL version="1.0"
LABEL description="Pap Smear Classification Pipeline"

WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required by OpenCV
# This is done in its own layer early on, as it rarely changes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the files required to install dependencies.
# This layer will only be rebuilt if these files change.
COPY pyproject.toml requirements.txt requirements-dev.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY create_fake_dataset.py .

RUN pip install --no-cache-dir -e .

# Set a default command to drop into a shell.
# This makes the container interactive for development and running scripts.
CMD ["bash"]