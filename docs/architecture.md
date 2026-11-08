# SmartDemand AI — System Architecture

## 1. Project overview

SmartDemand AI is a daily store-item demand forecasting system built around an XGBoost regression model.

It combines historical sales data, time-based feature engineering, recursive forecasting, a FastAPI inference service, AWS-hosted model artifacts, container deployment, and production monitoring.

The system supports forecasts for an individual store-item pair for a requested horizon of 1–90 days.

## 2. High-level architecture

The system has five main layers:

1. Data preparation and model development.
2. Amazon S3 artifact storage.
3. GitHub Actions and Amazon ECR for container delivery.
4. Amazon EC2 hosting the Dockerized FastAPI service.
5. Prometheus, Grafana, and Alertmanager for monitoring and email alerts.

## 3. Data and model pipeline

The model development workflow uses historical store-item sales data.

1. Validate the source data and inspect its quality.
2. Perform exploratory data analysis to understand sales trends,
   seasonality, store performance, and item demand.
3. Generate calendar, lag, and rolling-window features.
4. Train baseline and XGBoost models.
5. Evaluate models using a chronological 90-day validation period.
6. Compare MAE, RMSE, and RMSLE.
7. Train the final model and prepare the serving history.

Feature engineering includes calendar variables, lag features
(7, 14, 28, and 365 days), and shifted rolling means
(7, 14, and 30 days).

The rolling features are shifted to avoid using the current target value
when constructing historical training features.

## 4. Model artifacts and storage

The trained model and serving history are stored in Amazon S3.

| Artifact | S3 object key |
|---|---|
| Final XGBoost model | models/xgboost_final_model.joblib |
| Serving history | serving/history.csv |

The inference service downloads these artifacts when required and
uses the configured AWS region and S3 bucket.

The EC2 instance uses its IAM role for AWS access rather than embedding
AWS access keys in the application source code.

## 5. Forecast inference workflow

For each forecast request, the API:

1. Validates the store, item, and forecast horizon.
2. Loads the trained model and historical serving data.
3. Selects the requested store-item history.
4. Builds calendar, lag, and rolling-window features.
5. Predicts the next day's demand.
6. Adds the prediction to the history used for subsequent forecast days.
7. Repeats the process recursively until the requested horizon is reached.
8. Returns the forecast dates and predicted sales.

Predictions are clipped to non-negative values.

## 6. FastAPI service

The inference service is implemented with FastAPI and runs inside Docker
on the EC2 instance.

| Endpoint | Method | Purpose | Authentication |
|---|---|---|---|
| /health | GET | Reports API and model health | None |
| /metrics | GET | Exposes Prometheus metrics | Bearer token |
| /forecast | POST | Generates store-item demand forecasts | X-API-Key |

The forecast request accepts:

| Field | Validation | Default |
|---|---|---|
| store | Integer, 1–10 | Required |
| item | Integer, 1–50 | Required |
| horizon_days | Integer, 1–90 | 7 |

The API uses separate credentials for forecast requests and metrics
scraping. Production configuration disables interactive API documentation.

## 7. Container build and CI/CD

The application image is built using `deployment/Dockerfile`.

The image uses Python 3.13 slim, installs the API dependencies,
copies the application source, and runs Uvicorn as a non-root user.

GitHub Actions performs the following workflow:

1. Installs dependencies and runs the pytest suite.
2. Builds the Docker image and performs an import smoke test.
3. On pushes to the main branch, authenticates to Amazon ECR and
   publishes images tagged with the commit SHA and `latest`.

The current workflow publishes the image to ECR. It does not automatically
replace the running EC2 API container.

## 8. AWS runtime architecture

The production API runs as a Docker container on Amazon EC2.

- Amazon ECR stores the application image.
- Amazon S3 stores the model and serving history.
- The EC2 IAM role provides AWS permissions to the instance.
- Docker restart policies are configured to restart containers
  unless they are explicitly stopped.
- CloudWatch logging is configured for the API deployment.

The monitoring stack runs separately from the API container.

Operational caution: review `deployment/deploy.sh` before using it.
Its current cleanup command can remove every Docker container on the
instance, including monitoring services. Do not execute it until corrected.

## 9. Monitoring and alerting

The production API is monitored using Prometheus, Grafana, and Alertmanager.

- Prometheus scrapes the API metrics endpoint every 15 seconds.
- Grafana visualizes request rate, latency, error ratio, model readiness,
  forecast activity, API availability, CPU, and memory.
- Alertmanager routes configured alerts to an email receiver.

Configured alert rules include:

| Alert | Condition | Duration |
|---|---|---|
| SmartDemandApiUnreachable | API target is down | 2 minutes |
| SmartDemandApiModelNotLoaded | Model-loaded metric is zero | 2 minutes |
| SmartDemandApiHighServerErrorRate | 5xx error ratio exceeds 5% | 5 minutes |
| SmartDemandApiHighForecastLatency | Forecast p95 latency exceeds 2 seconds | 5 minutes |

Monitoring services bind to localhost ports on the EC2 host.
Persistent Docker volumes retain Prometheus, Grafana, and Alertmanager data.
Email notification delivery has been tested successfully.

## 10. Repository structure

Important project directories and files:

| Path | Responsibility |
|---|---|
| api/ | FastAPI application and routes |
| src/data/ | Data validation, EDA, and serving-history preparation |
| src/features/ | Feature engineering |
| src/models/ | Model training and evaluation |
| src/inference/ | Forecast inference logic |
| src/tracking/ | Experiment tracking |
| tests/ | API and inference tests |
| deployment/ | Docker and deployment configuration |
| deployment/monitoring/ | Prometheus, Grafana, and Alertmanager |
| docs/ | Project documentation |
| .github/workflows/ | CI and ECR publishing workflow |

## 11. Operational considerations

- Keep API keys, SMTP credentials, and other secrets outside source control.
- Use IAM roles for AWS access from EC2.
- Keep monitoring dashboards and alerting services separate from the API.
- Review deployment scripts before execution.
- Validate model and serving-history compatibility before deployment.
- Monitor forecast latency, API errors, and model availability.
- Confirm backups and persistent-volume health during maintenance.
