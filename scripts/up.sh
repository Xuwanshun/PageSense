#!/bin/bash
# =============================================================================
# up.sh — Start the application (scale ECS to 2 tasks)
# =============================================================================
# Fargate only charges when containers are running.
# Run this when you want to test or use the app.
# Takes ~2-3 minutes for containers to start and pass health checks.
#
# HOW TO RUN:
#   chmod +x scripts/up.sh
#   ./scripts/up.sh
# =============================================================================

set -e

REGION="${AWS_REGION:-ca-central-1}"

echo "Fetching stack outputs from CloudFormation..."

CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsClusterName'].OutputValue" \
  --output text --region "$REGION")

SERVICE=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsServiceName'].OutputValue" \
  --output text --region "$REGION")

APP_URL=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='AppUrl'].OutputValue" \
  --output text --region "$REGION")

DB_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name RagAgentDatabase \
  --query "Stacks[0].Outputs[?OutputKey=='DbEndpoint'].OutputValue" \
  --output text --region "$REGION")

# DB identifier is the hostname up to the first dot
DB_INSTANCE="${DB_ENDPOINT%%.*}"

# Start RDS if it's stopped
DB_STATUS=$(aws rds describe-db-instances \
  --db-instance-identifier "$DB_INSTANCE" \
  --query "DBInstances[0].DBInstanceStatus" \
  --output text --region "$REGION")

if [ "$DB_STATUS" = "stopped" ]; then
  echo "Starting RDS instance $DB_INSTANCE (was stopped)..."
  aws rds start-db-instance \
    --db-instance-identifier "$DB_INSTANCE" \
    --region "$REGION" --output none
  echo "RDS starting — waiting until available (3-5 min)..."
  aws rds wait db-instance-available \
    --db-instance-identifier "$DB_INSTANCE" \
    --region "$REGION"
  echo "RDS available."
elif [ "$DB_STATUS" = "available" ]; then
  echo "RDS already running."
else
  echo "RDS status: $DB_STATUS — proceeding anyway."
fi

echo "Scaling up ECS: cluster=$CLUSTER service=$SERVICE"

aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --desired-count 2 \
  --region "$REGION" \
  --output text --query "service.desiredCount"

echo ""
echo "Containers starting — takes about 2-3 minutes after RDS is up."
echo "Watch status: aws ecs describe-services --cluster $CLUSTER --services $SERVICE --region $REGION --query 'services[0].runningCount'"
echo ""
echo "Your app will be available at: $APP_URL"
