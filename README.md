# SmartDemand AI

SmartDemand AI forecasts daily item sales for a store. The FastAPI service uses
the trained XGBoost model and recent sales history to produce recursive
forecasts.

## Kaggle result

The recursive XGBoost submission for the Store Item Demand Forecasting
Challenge contained 45,000 rows and passed the local checks for columns,
missing values, duplicate IDs, test ID order, and nonnegative predictions.

| Leaderboard | Score |
| --- | ---: |
| Public | 14.37320 |
| Private | 12.95266 |

These are the scores reported after submission. The validation metrics in
`reports/xgboost/metrics.csv` are measured on the separate 90-day holdout and
should not be compared directly with the Kaggle leaderboard score.

## Run locally

Use Python 3.13. Install the API dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-api.txt
```

The service first looks for local artifacts at
`models/xgboost_final_model.joblib` and `data/serving/history.csv`. To load the
artifacts from S3 instead, set `S3_BUCKET`; the default object keys are
`models/xgboost_final_model.joblib` and `serving/history.csv`. Override them
with `MODEL_KEY` and `HISTORY_KEY` if needed. AWS credentials must be available
through the standard AWS credential chain.

Start the API:

```powershell
uvicorn api.main:app --reload
```

The service listens on `http://127.0.0.1:8000`. Interactive API documentation
is available at `/docs`.

## API

`GET /health` reports whether the model and history loaded during application
startup.

`GET /metrics` exposes Prometheus metrics for request counts and latency, model
readiness, requested forecast horizons, and predicted sales distributions. The
HTTP metrics use the route template (rather than store or item IDs) to keep
label cardinality bounded.

## Local monitoring stack

Start the API in one terminal after configuring its local or S3 artifacts:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In another PowerShell terminal, set a local Grafana admin password and start
Prometheus and Grafana:

```powershell
$env:GRAFANA_ADMIN_PASSWORD = "choose-a-local-password"
docker compose -f deployment/monitoring/compose.yaml up -d
```

Open Prometheus at `http://localhost:9090` to check the API target and alert
states. Open Grafana at `http://localhost:3000` and sign in as `admin` with the
password set above; the SmartDemand API dashboard and Prometheus data source
are provisioned automatically. The local Prometheus instance keeps seven days
of metrics. Its alert rules cover scrape failures, model readiness, and
elevated 5xx rates; notifications require a separately configured Alertmanager
receiver.

Stop the local monitoring containers with:

```powershell
docker compose -f deployment/monitoring/compose.yaml down
```

`POST /forecast` accepts a store, item, and forecast horizon:

```json
{
  "store": 1,
  "item": 1,
  "horizon_days": 7
}
```

Store IDs must be from 1 to 10, item IDs from 1 to 50, and the horizon from 1
to 90 days. The response includes one date and predicted sales value for each
forecast day. Requests for store/item pairs absent from the serving history
return HTTP 404.

## Build and run the container

From the repository root:

```powershell
docker build -f deployment/Dockerfile -t smartdemand-api .
docker run --rm -p 8000:8000 `
  -e S3_BUCKET=<your-artifact-bucket> `
  -e AWS_REGION=ap-southeast-2 `
  smartdemand-api
```

The container uses the AWS credential chain. In AWS, give its task or workload
role read access to the two artifact objects. The repository CI workflow runs
the API tests, builds and smoke-checks the image, then pushes `smartdemand-api`
to ECR on pushes to `main`. The ECR repository must already exist, and the
GitHub Actions secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` must be
configured for that account. The workflow publishes both the commit SHA tag
and `latest`; it does not roll out a running service.

## AWS runtime deployment

An ECS Express Mode deployment script and setup guide are available in
`deployment/ecs-express/`. The script requires explicit network settings,
defaults to private ingress, caps the service at one task, and supports
PowerShell `-WhatIf`. The API has no authentication, so review the setup guide
and AWS costs before creating a service; nothing in CI creates or updates the
service.

## Project layout

- `api/` — FastAPI application
- `src/inference/` — feature creation and recursive forecasting
- `src/data/` — data preparation, including serving-history generation
- `src/models/` — model training and validation scripts
- `deployment/` — API image, local monitoring stack, and guarded ECS Express Mode deployment prep
- `.github/workflows/ci.yml` — tests, image build, and ECR publishing
