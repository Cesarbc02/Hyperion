FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock ./

RUN uv sync --no-dev

COPY main.py helpers.py rag.py build_index.py ./
COPY rag_index ./rag_index
COPY knowledge ./knowledge

RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV IDE_BACKEND_URL=http://host.docker.internal:3001/api

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "main.py"]