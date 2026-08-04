FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY main.py ./

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "main.py"]
