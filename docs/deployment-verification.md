# Deployment verification

Deployed on AWS EC2 (ap-southeast-2) from the Docker image built by GitHub Actions and stored in ECR.
The model and serving history load from S3 at startup using an IAM instance role (no keys on the server).
Verified on 2026-11-03. Server address omitted.

## GET /health

{
  "status": "ok",
  "model_loaded": true
}

## POST /forecast  (store 1, item 1, 7 days)

{
  "store": 1,
  "item": 1,
  "horizon_days": 7,
  "predictions": [
    {
      "date": "2018-01-01",
      "predicted_sales": 12.6
    },
    {
      "date": "2018-01-02",
      "predicted_sales": 15.07
    },
    {
      "date": "2018-01-03",
      "predicted_sales": 14.87
    },
    {
      "date": "2018-01-04",
      "predicted_sales": 15.64
    },
    {
      "date": "2018-01-05",
      "predicted_sales": 16.55
    },
    {
      "date": "2018-01-06",
      "predicted_sales": 18.91
    },
    {
      "date": "2018-01-07",
      "predicted_sales": 18.86
    }
  ]
}

The seven values match the batch pipeline that produced the Kaggle submission (private SMAPE 12.95).
