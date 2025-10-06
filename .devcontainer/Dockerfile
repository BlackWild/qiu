# Base image
FROM ghcr.io/astral-sh/uv:0.8-python3.13-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# ENV HOME=/home/vscode

# Use the shared volume /home/vscode/ that is created by Docker when using vscode and dev containers for a persistent cache directory. The cache does not die when the container is rebuilt.
ENV UV_CACHE_DIR=/vscode/.cache/uv
# Prefer using Python versions managed by uv
ENV UV_PYTHON_PREFERENCE=only-managed
# Use copy mode for uv link to avoid issues with bind mounts on some systems
# ENV UV_LINK_MODE=copy

ENV RUFF_CACHE_DIR=/vscode/.cache/ruff
# ENV MYPY_CACHE_DIR=/home/vscode/.cache/mypy
# ENV PRE_COMMIT_HOME=/home/vscode/.cache/pre-commit
# ENV PYENV_ROOT=/home/vscode/.pyenv
# ENV PIPX_HOME=/home/vscode/.pipx



# Install git
RUN apt-get update && \
  apt-get install -y git

# Clean up
RUN apt-get clean && \
  rm -rf /var/lib/apt/lists/*
