FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and source code
COPY pyproject.toml .
COPY src/ ./src/
COPY config/ ./config/

# Build wheel
RUN pip install build && python -m build --wheel

FROM python:3.11-slim as runtime

WORKDIR /app

# Copy the built wheel from builder
COPY --from=builder /build/dist/*.whl .

# Install the wheel
RUN pip install *.whl && rm *.whl

# Expose FastAPI port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
