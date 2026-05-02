# api — Django REST container for Cloud Run

A single Django + DRF service that exposes both in-repo modules over HTTP:

- `eodhd_api` — every public method on `eodhd.EODHDClient`, mounted under `/eodhd/`.
- `timesig_api` — `time_signature.analyze_url`, mounted under `/timesig/`, with each run persisted to a Postgres `AnalysisRequest` row.

State lives in **Cloud SQL Postgres**; the container is **stateless** and ready for Cloud Run.

---

## Endpoints

| Method | Path                                | Notes                                                       |
|--------|-------------------------------------|-------------------------------------------------------------|
| GET    | `/health`                           | Liveness probe. Returns `{"status":"ok"}`.                  |
| GET    | `/eodhd/`                           | Lists every EODHD slug the dispatcher exposes.              |
| GET    | `/eodhd/<slug>[/<path-arg>]`        | Forwards query string to the matching `EODHDClient` method. |
| POST   | `/timesig/analyze`                  | Body: `{"url": "...", "backend?": "librosa", ...}`.         |
| GET    | `/timesig/analyses`                 | Past analyses, newest first.                                |
| GET    | `/timesig/analyses/<id>`            | One past analysis.                                          |

**EODHD examples**

```bash
# end-of-day for AAPL
curl 'https://<service>.run.app/eodhd/eod/AAPL.US?from=2024-01-01&to=2024-02-01'

# real-time quote
curl 'https://<service>.run.app/eodhd/real-time/AAPL.US'

# news for a ticker
curl 'https://<service>.run.app/eodhd/news?s=AAPL.US&limit=5'

# upcoming earnings
curl 'https://<service>.run.app/eodhd/calendar/earnings?from=2024-01-01&to=2024-01-31'
```

`from` is auto-translated to the client's `from_` kwarg. Numeric query values are coerced to `int`/`float` automatically.

**time_signature example**

```bash
curl -X POST 'https://<service>.run.app/timesig/analyze' \
     -H 'content-type: application/json' \
     -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

---

## Required environment variables

| Var                         | Purpose                                                                    |
|-----------------------------|----------------------------------------------------------------------------|
| `DJANGO_SECRET_KEY`         | Django secret key. Required in production.                                 |
| `DJANGO_ALLOWED_HOSTS`      | Comma-separated; defaults to `.run.app,localhost,127.0.0.1`.               |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | Comma-separated full URLs (e.g. `https://my-svc-xyz.run.app`).            |
| `DB_NAME` / `DB_USER` / `DB_PASSWORD` | Cloud SQL Postgres credentials.                                  |
| `INSTANCE_CONNECTION_NAME`  | `<project>:<region>:<instance>` — switches DB host to the Unix socket.     |
| `EODHD_API_TOKEN`           | EODHD API key. Required for `/eodhd/*` endpoints to return data.           |
| `TIMESIG_MAX_DURATION`      | Seconds of audio analyzed per request when caller doesn't pass `end_time`. |

If `INSTANCE_CONNECTION_NAME` is unset but `DB_NAME` is, the app connects via TCP using `DB_HOST`/`DB_PORT` (good for the cloud-sql-auth-proxy locally). With no `DB_NAME`, it falls back to SQLite for `manage.py check`.

---

## Local dev

```bash
cd api
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt

# Optional: madmom backend. requirements.txt deliberately doesn't list madmom
# directly because it can't survive pip's build isolation; install it as a
# follow-on step:
pip install --no-build-isolation 'madmom>=0.16.1'

# point Python at the in-repo modules
export PYTHONPATH="$(pwd)/..:$(pwd)/../eodhd:$(pwd)/../time_signature"

export DJANGO_DEBUG=1
export EODHD_API_TOKEN=demo
python manage.py migrate
python manage.py runserver
```

`PYTHONPATH` only matters locally; the Dockerfile copies the packages into `/app` so they're importable directly inside the container.

---

## Build & deploy to Cloud Run

Run from the **repo root** (the Dockerfile expects that build context):

```bash
PROJECT_ID=your-gcp-project
REGION=us-central1
SERVICE=timesig-eodhd-api
INSTANCE=$PROJECT_ID:$REGION:timesig-pg

# 1. Cloud SQL Postgres instance + database (one-time)
gcloud sql instances create timesig-pg \
    --database-version=POSTGRES_16 \
    --tier=db-f1-micro \
    --region=$REGION
gcloud sql databases create appdb --instance=timesig-pg
gcloud sql users create appuser --instance=timesig-pg --password='CHANGE-ME'

# 2. Store secrets (recommended over passing on the CLI)
printf 'CHANGE-ME' | gcloud secrets create db-password --data-file=-
printf 'your-eodhd-token' | gcloud secrets create eodhd-token --data-file=-
python -c "import secrets; print(secrets.token_urlsafe(50))" \
    | gcloud secrets create django-secret --data-file=-

# 3. Build the image with Cloud Build
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$SERVICE \
    --file api/Dockerfile \
    .

# 4. Deploy. --add-cloudsql-instances mounts the socket at
#    /cloudsql/$INSTANCE which settings.py picks up automatically.
gcloud run deploy $SERVICE \
    --image gcr.io/$PROJECT_ID/$SERVICE \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --add-cloudsql-instances $INSTANCE \
    --memory 4Gi \
    --cpu 2 \
    --timeout 600 \
    --concurrency 4 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars "INSTANCE_CONNECTION_NAME=$INSTANCE,DB_NAME=appdb,DB_USER=appuser,TIMESIG_MAX_DURATION=60" \
    --set-secrets "DB_PASSWORD=db-password:latest,EODHD_API_TOKEN=eodhd-token:latest,DJANGO_SECRET_KEY=django-secret:latest"

# 5. Run migrations once (Cloud Run jobs work nicely for this; quickest path is exec):
gcloud run jobs create $SERVICE-migrate \
    --image gcr.io/$PROJECT_ID/$SERVICE \
    --region $REGION \
    --add-cloudsql-instances $INSTANCE \
    --set-env-vars "INSTANCE_CONNECTION_NAME=$INSTANCE,DB_NAME=appdb,DB_USER=appuser" \
    --set-secrets "DB_PASSWORD=db-password:latest,DJANGO_SECRET_KEY=django-secret:latest" \
    --command python --args manage.py,migrate
gcloud run jobs execute $SERVICE-migrate --region $REGION --wait
```

After the first successful deploy, grab the URL and add it to `DJANGO_CSRF_TRUSTED_ORIGINS` if you ever serve unsafe POSTs from a browser.

---

## Notes worth knowing before shipping

- **Image size is ~2 GB.** TensorFlow + librosa + ffmpeg dominate. Cloud Run cold starts will be 20–40 s on first hit; set `--min-instances 1` if that's unacceptable.
- **`/timesig/analyze` is synchronous.** A YouTube clip can take 30 s+ to download and analyze. The deploy command above sets `--timeout 600` so a single request can run up to 10 minutes. For real traffic, move the analysis to Cloud Tasks + a worker and have the endpoint return `202 Accepted` with the row id.
- **No auth.** DRF is configured `AllowAny`. Put Cloud Run IAM auth in front, or add a real auth class, before exposing this externally.
- **YAMNet downloads on first call.** First `/timesig/analyze` after a cold start will fetch the model from `tfhub.dev` (~15 MB) and cache it for the life of the instance.
