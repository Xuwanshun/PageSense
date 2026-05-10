#!/bin/bash
# =============================================================================
# down.sh — Stop the application (scale ECS to 0 tasks)
# =============================================================================
# Scaling to 0 stops all Fargate containers. Fargate cost drops to $0.
# The ALB, RDS, VPC, and S3 stay running (small fixed cost ~$25-37/month).
# All your data is preserved in RDS and S3.
#
# HOW TO RUN:
#   chmod +x scripts/down.sh
#   ./scripts/down.sh
# =============================================================================

set -e

REGION="${AWS_REGION:-ca-central-1}"

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

echo "Scaling down to 0: cluster=$CLUSTER service=$SERVICE"

aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --desired-count 0 \
  --region "$REGION" \
  --output text --query "service.desiredCount"

echo ""
echo "Containers stopping. Fargate cost is now \$0."
echo "Your data is safe in RDS and S3."
echo "Run ./scripts/up.sh when you want to start again."
