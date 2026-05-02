"""
Django settings for the api container.

All environment-driven config is read from process env vars so the same image
runs locally (with SQLite or a local Postgres) and on Cloud Run (with Cloud
SQL via Unix socket).
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = _env_bool("DJANGO_DEBUG", False)

# Cloud Run sets the K_SERVICE env var; we trust the *.run.app host that routes
# requests to us. For local dev, allow everything when DEBUG.
ALLOWED_HOSTS = (
    ["*"]
    if DEBUG
    else [h.strip() for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",") if h.strip()]
)
if not ALLOWED_HOSTS:
    ALLOWED_HOSTS = [".run.app", "localhost", "127.0.0.1"]

# Cloud Run terminates TLS upstream and forwards X-Forwarded-Proto.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

# CSRF: Cloud Run service URLs are required for any unsafe POSTs from a browser.
CSRF_TRUSTED_ORIGINS = [
    o.strip()
    for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if o.strip()
]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "rest_framework",
    "eodhd_api",
    "timesig_api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "core.urls"
WSGI_APPLICATION = "core.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    },
]


# ---------------------------------------------------------------- database
#
# Cloud SQL Postgres connection. On Cloud Run, mount the instance via the
# built-in Cloud SQL connector by passing --add-cloudsql-instances=<conn>;
# the socket appears at /cloudsql/<INSTANCE_CONNECTION_NAME>. Locally, set
# DB_HOST to a TCP host (e.g. via the cloud-sql-auth-proxy) instead.

_db_name = os.environ.get("DB_NAME", "")
_db_user = os.environ.get("DB_USER", "")
_db_password = os.environ.get("DB_PASSWORD", "")
_instance_conn = os.environ.get("INSTANCE_CONNECTION_NAME", "")

if _db_name:
    if _instance_conn:
        DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": _db_name,
                "USER": _db_user,
                "PASSWORD": _db_password,
                "HOST": f"/cloudsql/{_instance_conn}",
                "PORT": "",
            }
        }
    else:
        DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": _db_name,
                "USER": _db_user,
                "PASSWORD": _db_password,
                "HOST": os.environ.get("DB_HOST", "127.0.0.1"),
                "PORT": os.environ.get("DB_PORT", "5432"),
            }
        }
else:
    # Fallback for `manage.py check` and local prototyping with no DB configured.
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }


# ---------------------------------------------------------------- DRF

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
}


# ---------------------------------------------------------------- i18n / static

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ---------------------------------------------------------------- module config

# EODHD API token; required for any /eodhd/* endpoint to actually return data.
EODHD_API_TOKEN = os.environ.get("EODHD_API_TOKEN", "")

# Cap analysis runtime so a single request can't tie up a Cloud Run instance.
TIMESIG_DEFAULT_MAX_DURATION = float(os.environ.get("TIMESIG_MAX_DURATION", "60"))
