FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

# For multi stage builds,
# create a venv in the first stage, install deps there and 
# copy the entire venv to the second stage. This is a clean approach
# for multi stage python imaages.

COPY src/__init__.py src/features.py src/
COPY app/ app/
COPY params.yaml metrics.json ./
# The model is baked into the image at build time (dvc pull runs before docker build),
# so every image tag maps to exactly one model.
COPY models/model.json models/model.json

# Set to the release tag by release.yml, shown by /model-info and /predict.
ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

# Create a non-root system user to run the application
RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --shell /usr/sbin/nologin appuser

# Entire application code stays owned by root, so that the user can't modify it
# even if the container gets compromised.
USER 10001:10001

# For informational purpose only
EXPOSE  8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
