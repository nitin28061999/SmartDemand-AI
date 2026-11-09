#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

REGION="ap-southeast-2"
BUCKET="smartdemand-ai-mlops-nitin-2026-7f3a9c"
CONTAINER="smartdemand-api"
CANDIDATE="smartdemand-api-candidate"
BACKUP="smartdemand-api-previous"
LOG_GROUP="/smartdemand/api"
METRICS_SECRET_FILE="deployment/monitoring/secrets/metrics_api_key"
CANDIDATE_PORT="18001"
PRODUCTION_PORT="8000"


  # Verify required commands are available.
for cmd in aws sudo docker curl grep seq sleep; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $cmd"
        exit 1
    fi
done

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${ECR_REGISTRY}/smartdemand-api:latest"

# Verify the CloudWatch log group exists before starting containers.
LOG_GROUPS="$(aws logs describe-log-groups \
    --log-group-name-prefix "$LOG_GROUP" \
    --query 'logGroups[].logGroupName' \
    --output text \
    --region "$REGION")"

if ! grep -Fxq "$LOG_GROUP" <<< "$LOG_GROUPS"; then
    echo "ERROR: CloudWatch log group $LOG_GROUP does not exist."
    exit 1
fi

# Retrieve secrets without printing them.
API_KEY="$(aws ssm get-parameter \
    --name /smartdemand/api-key \
    --with-decryption \
    --query Parameter.Value \
    --output text \
    --region "$REGION")"

if [[ -z "$API_KEY" || "$API_KEY" == "None" ]]; then
    echo "ERROR: API key unavailable. Aborting."
    exit 1
fi

if ! sudo test -s "$METRICS_SECRET_FILE"; then
    echo "ERROR: Metrics secret file missing or empty."
    unset API_KEY
    exit 1
fi

METRICS_API_KEY="$(sudo cat "$METRICS_SECRET_FILE")"

if [[ -z "$METRICS_API_KEY" ]]; then
    echo "ERROR: Metrics API key unavailable."
    unset API_KEY
    exit 1
fi

HAD_OLD_CONTAINER=false
RECOVERY_ENABLED=false

# Refuse to overwrite an earlier backup or candidate container.
if sudo docker container inspect "$BACKUP" >/dev/null 2>&1; then
    echo "ERROR: Backup container $BACKUP already exists."
    echo "Inspect it before attempting another deployment."
    exit 1
fi

if sudo docker container inspect "$CANDIDATE" >/dev/null 2>&1; then
    echo "ERROR: Candidate container $CANDIDATE already exists."
    echo "Inspect it before attempting another deployment."
    exit 1
fi

# Pull the image before touching the running production container.
echo "Logging in to ECR and pulling the latest image..."

aws ecr get-login-password --region "$REGION" |
    sudo docker login --username AWS --password-stdin "$ECR_REGISTRY"

sudo docker pull "$IMAGE"

# Start a container with the same configuration used in production.
run_api_container() {
    local name="$1"
    local host_port="$2"
    local restart_policy="$3"

    sudo docker run -d \
        --name "$name" \
        --restart "$restart_policy" \
        -p "${host_port}:8000" \
        --read-only \
        --tmpfs /tmp:rw,size=128m \
        --cap-drop ALL \
        --security-opt no-new-privileges \
        --log-driver awslogs \
        --log-opt "awslogs-region=$REGION" \
        --log-opt "awslogs-group=$LOG_GROUP" \
        -e S3_BUCKET="$BUCKET" \
        -e AWS_REGION="$REGION" \
        -e ENVIRONMENT=production \
        -e ENABLE_DOCS=false \
        -e API_KEY="$API_KEY" \
        -e METRICS_API_KEY="$METRICS_API_KEY" \
        "$IMAGE"
}

# Check health and metrics authentication on a specified host port.
verify_api() {
    local port="$1"

    echo "Checking API health on port $port..."

    for attempt in $(seq 1 30); do
        if curl -fsS --max-time 3 \
            "http://127.0.0.1:${port}/health" 2>/dev/null |
            grep -q '"model_loaded":true'; then

            echo "API health check passed."
            break
        fi

        if [[ "$attempt" -eq 30 ]]; then
            echo "ERROR: API health check failed on port $port."
            return 1
        fi

        sleep 2
    done

    echo "Checking authenticated metrics endpoint..."

    if ! curl -fsS --max-time 5 \
        -H "Authorization: Bearer $METRICS_API_KEY" \
        "http://127.0.0.1:${port}/metrics" >/dev/null; then
        echo "ERROR: Metrics authentication check failed on port $port."
        return 1
    fi

    echo "Metrics authentication check passed."
    return 0
}

# Restore the old API if the script exits unexpectedly while replacing it.
deployment_exit_handler() {
    local exit_code="$1"

    if [[ "$exit_code" -ne 0 &&
          "$RECOVERY_ENABLED" == "true" &&
          "$HAD_OLD_CONTAINER" == "true" ]]; then
        if sudo docker container inspect "$BACKUP" >/dev/null 2>&1; then
            echo "Unexpected deployment exit. Attempting automatic recovery..."
            sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

            if sudo docker rename "$BACKUP" "$CONTAINER" &&
               sudo docker start "$CONTAINER"; then
                if verify_api "$PRODUCTION_PORT"; then
                    echo "RECOVERY SUCCESS: Previous API is healthy."
                else
                    echo "CRITICAL: Previous API could not be verified." >&2
                    sudo docker logs --tail 50 "$CONTAINER" || true
                fi
            else
                echo "CRITICAL: Automatic recovery failed." >&2
            fi
        elif sudo docker container inspect "$CONTAINER" >/dev/null 2>&1; then
            # The exit may have occurred after stopping the old container but
            # before renaming it to the rollback name.
            sudo docker start "$CONTAINER" >/dev/null 2>&1 || true
        fi
    fi

    unset API_KEY METRICS_API_KEY
    return "$exit_code"
}

trap 'deployment_exit_handler $?' EXIT

# Phase 1: Test the new image without disrupting production.
echo "Starting candidate container on port $CANDIDATE_PORT..."

if ! run_api_container "$CANDIDATE" "$CANDIDATE_PORT" "no"; then
    echo "ERROR: Candidate container failed to start."
    exit 1
fi

if ! verify_api "$CANDIDATE_PORT"; then
    sudo docker logs --tail 50 "$CANDIDATE" || true
    sudo docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true
    echo "Candidate verification failed. Existing production API was not changed."
    exit 1
fi

# Candidate passed; remove it before moving the production container.
sudo docker rm -f "$CANDIDATE" >/dev/null

# Phase 2: Preserve the old container before replacing it.
RECOVERY_ENABLED=true

if sudo docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    HAD_OLD_CONTAINER=true

    echo "Preserving the existing API container as $BACKUP..."

    if ! sudo docker rename "$CONTAINER" "$BACKUP"; then
        echo "ERROR: Could not rename the existing API container."
        exit 1
    fi

    if ! sudo docker stop "$BACKUP"; then
        echo "ERROR: Could not stop the old API container. Restoring its name."
        sudo docker rename "$BACKUP" "$CONTAINER" || true
        sudo docker start "$CONTAINER" || true
        exit 1
    fi
fi

# Restore the previous container if production verification fails.
rollback_deployment() {
    echo "Deployment failed. Attempting rollback..."

    sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

    if [[ "$HAD_OLD_CONTAINER" == "true" ]]; then
        if sudo docker container inspect "$BACKUP" >/dev/null 2>&1; then
            sudo docker rename "$BACKUP" "$CONTAINER"
            sudo docker start "$CONTAINER"

            if verify_api "$PRODUCTION_PORT"; then
                echo "ROLLBACK SUCCESS: Previous API is healthy."
                return 0
            fi

            echo "CRITICAL: Previous API was restored but failed verification."
            sudo docker logs --tail 50 "$CONTAINER" || true
            return 1
        fi
    fi

    echo "No previous container was available to restore."
    return 1
}

# Phase 3: Start the new production container.
echo "Starting new production API on port $PRODUCTION_PORT..."

if ! run_api_container "$CONTAINER" "$PRODUCTION_PORT" "unless-stopped"; then
    rollback_deployment || true
    exit 1
fi

if ! verify_api "$PRODUCTION_PORT"; then
    sudo docker logs --tail 50 "$CONTAINER" || true
    rollback_deployment || true
    exit 1
fi
# New production API is verified; disable automatic recovery.
RECOVERY_ENABLED=false

# Phase 4: Remove the backup only after successful production verification.
if [[ "$HAD_OLD_CONTAINER" == "true" ]]; then
    sudo docker rm -f "$BACKUP" >/dev/null
fi


unset API_KEY METRICS_API_KEY

echo "SUCCESS: New API deployment passed health and metrics checks."
echo "========== RUNNING CONTAINERS =========="
sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo "========== API CONTAINER USER =========="
sudo docker exec "$CONTAINER" id
