#!/bin/bash
set -euo pipefail
REGION=ap-southeast-2
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=smartdemand-ai-mlops-nitin-2026-7f3a9c
IMAGE=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/smartdemand-api:latest

API_KEY=$(aws ssm get-parameter --name /smartdemand/api-key --with-decryption --query Parameter.Value --output text --region $REGION)
[ -n "$API_KEY" ] || { echo "API key is empty, aborting"; exit 1; }

aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
sudo docker pull $IMAGE
sudo docker ps -aq | xargs -r sudo docker rm -f
sudo docker run -d --restart unless-stopped -p 8000:8000 \
  --read-only --tmpfs /tmp:rw,size=128m \
  --cap-drop ALL --security-opt no-new-privileges \
  --log-driver awslogs --log-opt awslogs-region=$REGION --log-opt awslogs-group=/smartdemand/api \
  -e S3_BUCKET=$BUCKET -e AWS_REGION=$REGION \
  -e ENVIRONMENT=production -e ENABLE_DOCS=false -e API_KEY="$API_KEY" \
  $IMAGE
unset API_KEY
sleep 25
sudo docker ps --format "table {{.Status}}\t{{.Ports}}"
sudo docker exec $(sudo docker ps -q) id