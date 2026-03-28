FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project metadata and lockfile first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile
RUN uv sync --frozen --no-dev

# Copy project source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install package in editable mode
RUN uv pip install -e . --no-deps

# Volumes for configs and checkpoints are mounted at runtime
VOLUME ["/app/configs", "/app/checkpoints", "/app/data"]

ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/server.yaml"]
