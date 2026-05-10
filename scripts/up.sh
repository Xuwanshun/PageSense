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

# Read the cluster and service names from CDK stack outputs.
# These were printed when you ran `cdk deploy RagAgentApp`.
echo "Fetching ECS cluster and service names from CloudFormation..."

CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsClusterName'].OutputValue" \
  --output text \
  --region "$REGION")

SERVICE=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsServiceName'].OutputValue" \
  --output text \
  --region "$REGION")

APP_URL=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='AppUrl'].OutputValue" \
  --output text \
  --region "$REGION")

echo "Scaling up: cluster=$CLUSTER service=$SERVICE"

aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --desired-count 2 \
  --region "$REGION" \
  --output text --query "service.desiredCount"

echo ""
echo "Containers starting — takes about 2-3 minutes."
echo "Watch status: aws ecs describe-services --cluster $CLUSTER --services $SERVICE --region $REGION --query 'services[0].runningCount'"
echo ""
echo "Your app will be available at: $APP_URL"
