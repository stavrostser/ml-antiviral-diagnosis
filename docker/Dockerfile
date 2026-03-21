FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    make \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Install dependencies first for better layer caching.
COPY pyproject.toml /workspace/
RUN uv sync --no-install-project

COPY . /workspace

ENV PATH="/workspace/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "ml_antiviral_diagnosis.api:app", "--host", "0.0.0.0", "--port", "8000"]
