#!/bin/bash
# =============================================================================
# create-secrets.sh
# =============================================================================
# Creates or updates secrets in AWS Secrets Manager.
# Safe to run multiple times — existing secrets are updated, not duplicated.
#
# Run BEFORE cdk deploy (except DATABASE_URL — skip that, fill after deploy).
#
# HOW TO RUN:
#   chmod +x scripts/create-secrets.sh
#   ./scripts/create-secrets.sh
# =============================================================================

set -e

REGION="${AWS_REGION:-ca-central-1}"

echo "Creating/updating secrets in AWS Secrets Manager (region: $REGION)"
echo ""

# Helper: create secret if it doesn't exist, update if it does
upsert_secret() {
  local name="$1"
  local description="$2"
  local value="$3"

  if aws secretsmanager describe-secret --secret-id "$name" --region "$REGION" &>/dev/null; then
    aws secretsmanager put-secret-value \
      --secret-id "$name" \
      --secret-string "$value" \
      --region "$REGION" > /dev/null
    echo "✓ $name updated"
  else
    aws secretsmanager create-secret \
      --name "$name" \
      --description "$description" \
      --secret-string "$value" \
      --region "$REGION" > /dev/null
    echo "✓ $name created"
  fi
}

# ── Secret 1: OpenAI API Key ──────────────────────────────────────────────────
read -p "Enter your OpenAI API key (sk-...): " OPENAI_KEY
upsert_secret "rag-agent/openai-api-key" "OpenAI API key for embeddings and LLM calls" "$OPENAI_KEY"

# ── Secret 2: JWT Secret Key ──────────────────────────────────────────────────
echo ""
JWT_KEY=$(openssl rand -hex 32)
echo "Generated JWT secret key (auto-generated, stored in Secrets Manager):"
echo "  $JWT_KEY"
upsert_secret "rag-agent/jwt-secret-key" "JWT signing secret for user authentication tokens" "$JWT_KEY"

# ── Secret 3: Database URL ────────────────────────────────────────────────────
# Skip this now — you need the RDS endpoint from CDK output first.
# After `cdk deploy RagAgentDatabase`, re-run this script or run:
#   ./scripts/set-database-url.sh
echo ""
echo "Database URL: skipping for now (needs RDS endpoint from CDK output)."
echo "  After cdk deploy, run: ./scripts/set-database-url.sh"

# ── Secret 4: Google OAuth credentials ───────────────────────────────────────
echo ""
read -p "Enter Google OAuth Client ID (leave blank to skip): " GOOGLE_CLIENT_ID
if [ -n "$GOOGLE_CLIENT_ID" ]; then
  read -p "Enter Google OAuth Client Secret: " GOOGLE_CLIENT_SECRET
  upsert_secret "rag-agent/google-client-id" "Google OAuth 2.0 client ID" "$GOOGLE_CLIENT_ID"
  upsert_secret "rag-agent/google-client-secret" "Google OAuth 2.0 client secret" "$GOOGLE_CLIENT_SECRET"
else
  echo "— Google OAuth skipped"
fi

echo ""
echo "Done. Next: cd cdk && cdk deploy RagAgentNetwork RagAgentDatabase"
echo "Then run: ./scripts/set-database-url.sh"
echo "Then run: cdk deploy RagAgentApp"
