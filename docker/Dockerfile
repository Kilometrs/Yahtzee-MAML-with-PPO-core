FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Install package in editable mode
RUN pip install -e .

# Volumes for configs and checkpoints are mounted at runtime
VOLUME ["/app/configs", "/app/checkpoints", "/app/data"]

ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/server.yaml"]
