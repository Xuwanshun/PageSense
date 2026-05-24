#!/bin/bash
# =============================================================================
# down.sh — Stop the application (scale ECS to 0 tasks)
# =============================================================================
# Scales ECS to 0 and stops RDS. Only the ALB keeps running (~$16-20/month).
# All your data is preserved in RDS (stopped, not deleted) and S3.
#
# HOW TO RUN:
#   chmod +x scripts/down.sh
#   ./scripts/down.sh
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

DB_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name RagAgentDatabase \
  --query "Stacks[0].Outputs[?OutputKey=='DbEndpoint'].OutputValue" \
  --output text --region "$REGION")

DB_INSTANCE="${DB_ENDPOINT%%.*}"

echo "Scaling down ECS to 0: cluster=$CLUSTER service=$SERVICE"

aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --desired-count 0 \
  --region "$REGION" \
  --output text --query "service.desiredCount"

# Stop RDS if running
DB_STATUS=$(aws rds describe-db-instances \
  --db-instance-identifier "$DB_INSTANCE" \
  --query "DBInstances[0].DBInstanceStatus" \
  --output text --region "$REGION")

if [ "$DB_STATUS" = "available" ]; then
  echo "Stopping RDS instance $DB_INSTANCE..."
  aws rds stop-db-instance \
    --db-instance-identifier "$DB_INSTANCE" \
    --region "$REGION" --output none
  echo "RDS stopping."
elif [ "$DB_STATUS" = "stopped" ]; then
  echo "RDS already stopped."
else
  echo "RDS status: $DB_STATUS — skipping stop."
fi

echo ""
echo "ECS and RDS are stopping. Cost is now ~\$16-20/month (ALB only)."
echo "Your data is safe in RDS and S3."
echo "Run ./scripts/up.sh when you want to start again."
