FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Pin UV to the conda-managed Python in this base image.
# pytorch/pytorch uses conda; UV_PYTHON ensures uv targets the correct interpreter
# rather than relying on PATH resolution, which may change across base image updates.
ENV UV_PYTHON=/opt/conda/bin/python

# Copy dependency manifest and lockfile first (layer cache optimisation).
# Changing src/ will not invalidate this layer.
COPY pyproject.toml uv.lock ./

# Install all runtime dependencies from the lockfile.
# --no-install-project: skip building the local package (src/ not copied yet).
# --frozen: never re-resolve; use uv.lock exactly as committed.
RUN uv sync --system --frozen --no-install-project

# Copy source and install the project package.
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN uv sync --system --frozen

VOLUME ["/app/configs", "/app/checkpoints", "/app/data"]

ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/server.yaml"]
