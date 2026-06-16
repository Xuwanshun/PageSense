#!/bin/bash
# =============================================================================
# set-flags.sh — Toggle RAG feature flags on the running ECS service
# =============================================================================
# Directly patches the ECS task definition environment variables and forces
# a new deployment — no Docker rebuild needed (~30 seconds vs ~20 minutes).
#
# Recommended off (high latency cost, marginal accuracy gain):
#   USE_LLM_RERANKER=false        saves ~1 LLM call per query
#   USE_CONTEXT_COMPRESSION=false saves ~1 LLM call per query
#   USE_FAITHFULNESS_CHECK=false  saves ~1-2 LLM calls per query
#
# HOW TO RUN:
#   chmod +x scripts/set-flags.sh
#   ./scripts/set-flags.sh
#
# To change the flags, edit the OVERRIDES section below.
# =============================================================================

set -e

REGION="${AWS_REGION:-ca-central-1}"

# ── Edit these to change which flags are on/off ──────────────────────────────
declare -A OVERRIDES=(
  # Keep on: big retrieval quality gains, zero or low query latency cost
  [USE_QUERY_ENHANCEMENT]="true"
  [USE_HYBRID_RETRIEVAL]="true"
  [USE_DOCUMENT_INTELLIGENCE]="true"
  [USE_ADAPTIVE_CHUNKING]="true"
  [USE_VLM_SUMMARIES]="true"

  # Turn off: each adds 1-2 sequential LLM API calls per query
  [USE_LLM_RERANKER]="false"
  [USE_CONTEXT_COMPRESSION]="false"
  [USE_FAITHFULNESS_CHECK]="false"
)
# ─────────────────────────────────────────────────────────────────────────────

echo "Fetching ECS cluster and service names..."
CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsClusterName'].OutputValue" \
  --output text --region "$REGION")

SERVICE=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsServiceName'].OutputValue" \
  --output text --region "$REGION")

echo "Cluster: $CLUSTER"
echo "Service: $SERVICE"

# Get the ARN of the currently active task definition
TASK_DEF_ARN=$(aws ecs describe-services \
  --cluster "$CLUSTER" \
  --services "$SERVICE" \
  --query "services[0].taskDefinition" \
  --output text --region "$REGION")

echo "Current task definition: $TASK_DEF_ARN"

# Download it as JSON
aws ecs describe-task-definition \
  --task-definition "$TASK_DEF_ARN" \
  --query taskDefinition \
  --region "$REGION" \
  > /tmp/task-def.json

# Apply each override: update existing env var or append a new one
PATCHED=/tmp/task-def-patched.json
cp /tmp/task-def.json "$PATCHED"

for KEY in "${!OVERRIDES[@]}"; do
  VALUE="${OVERRIDES[$KEY]}"
  echo "  Setting $KEY=$VALUE"

  # If the key already exists, replace its value; otherwise append it
  if python3 -c "
import json, sys
data = json.load(open('$PATCHED'))
env = data['containerDefinitions'][0]['environment']
found = False
for e in env:
    if e['name'] == '$KEY':
        e['value'] = '$VALUE'
        found = True
        break
if not found:
    env.append({'name': '$KEY', 'value': '$VALUE'})
json.dump(data, open('$PATCHED', 'w'), indent=2)
"; then
    : # success
  else
    echo "ERROR: failed to patch $KEY"
    exit 1
  fi
done

# Strip fields that must not be present when registering a new revision
REGISTER_JSON=$(python3 -c "
import json
data = json.load(open('$PATCHED'))
for field in ['taskDefinitionArn', 'revision', 'status', 'requiresAttributes',
              'compatibilities', 'registeredAt', 'registeredBy', 'deregisteredAt']:
    data.pop(field, None)
print(json.dumps(data, indent=2))
")

echo ""
echo "Registering new task definition revision..."
NEW_ARN=$(echo "$REGISTER_JSON" | aws ecs register-task-definition \
  --cli-input-json file:///dev/stdin \
  --query "taskDefinition.taskDefinitionArn" \
  --output text --region "$REGION")

echo "New task definition: $NEW_ARN"

echo ""
echo "Forcing new ECS deployment with updated flags..."
aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --task-definition "$NEW_ARN" \
  --force-new-deployment \
  --region "$REGION" \
  --output text --query "service.taskDefinition" > /dev/null

echo ""
echo "Done. New containers rolling out (~2-3 min)."
echo "Current flags applied:"
for KEY in "${!OVERRIDES[@]}"; do
  echo "  $KEY=${OVERRIDES[$KEY]}"
done
echo ""
echo "Watch rollout:"
echo "  aws ecs describe-services --cluster $CLUSTER --services $SERVICE --region $REGION --query 'services[0].{running:runningCount,pending:pendingCount,desired:desiredCount}'"
