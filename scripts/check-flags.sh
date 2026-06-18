#!/bin/bash
# =============================================================================
# check-flags.sh — Show current RAG feature flag values on the ECS service
# =============================================================================
# HOW TO RUN:
#   chmod +x scripts/check-flags.sh
#   ./scripts/check-flags.sh
# =============================================================================

set -e

REGION="${AWS_REGION:-ca-central-1}"

CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsClusterName'].OutputValue" \
  --output text --region "$REGION")

SERVICE=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsServiceName'].OutputValue" \
  --output text --region "$REGION")

TASK_DEF_ARN=$(aws ecs describe-services \
  --cluster "$CLUSTER" \
  --services "$SERVICE" \
  --query "services[0].taskDefinition" \
  --output text --region "$REGION")

echo "Task definition: $TASK_DEF_ARN"
echo ""

aws ecs describe-task-definition \
  --task-definition "$TASK_DEF_ARN" \
  --region "$REGION" \
  --query "taskDefinition.containerDefinitions[0].environment[?contains(name, 'USE_') || contains(name, 'PREFER_')]" \
  --output table
