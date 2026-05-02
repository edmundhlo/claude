# syntax=docker/dockerfile:1.7
#
# Build context = repo root (so we can pull in the eodhd/ and time_signature/
# packages alongside this api/ directory). From the repo root:
#
#     docker build -f api/Dockerfile -t timesig-eodhd-api:latest .
#
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DJANGO_SETTINGS_MODULE=core.settings \
    PORT=8080

# System deps:
#   ffmpeg          -> yt-dlp uses it to extract audio
#   libsndfile1     -> librosa/soundfile decoders
#   libpq5          -> runtime lib for psycopg
#   ca-certificates -> outbound HTTPS to EODHD / YouTube / Google APIs
#   build-essential -> compiler toolchain for madmom (no manylinux wheel for cp312)
#   python3-dev     -> Python headers required by madmom's C extensions
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libpq5 \
        ca-certificates \
        build-essential \
        python3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps + madmom in one layer so we can purge the build toolchain
# afterwards (saves ~250 MB in the final image).
#
# madmom is installed separately because pip's build isolation hides Cython +
# numpy from its setup.py; requirements.txt installs Cython into the venv first.
COPY api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
 && pip install --no-build-isolation 'madmom>=0.16.1' \
 && apt-get purge -y --auto-remove build-essential python3-dev \
 && rm -rf /var/lib/apt/lists/*

# In-repo packages: import as `eodhd` and `time_signature` from /app.
COPY eodhd/eodhd /app/eodhd
COPY time_signature/time_signature /app/time_signature

# Django project itself.
COPY api /app

EXPOSE 8080

# Single worker keeps RAM bounded (TensorFlow/YAMNet is heavy).
# Threads handle EODHD passthrough concurrency. timeout=0 lets long
# time_signature analyses run; raise the Cloud Run request timeout to match
# (`--timeout=600s` on `gcloud run deploy`).
CMD ["sh", "-c", "exec gunicorn core.wsgi:application --bind :$PORT --workers 1 --threads 8 --timeout 0"]
