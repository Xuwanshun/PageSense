#!/bin/bash
# Polls quota approval status and runs terraform apply when approved.
# Usage: bash launch_when_ready.sh
# Safe to Ctrl+C and re-run — terraform apply is idempotent.

set -euo pipefail

SPOT_CODE="L-3819A6DF"
REGION="us-east-1"
POLL_INTERVAL=60

echo "Checking GPU quota status..."

while true; do
    SPOT_STATUS=$(aws service-quotas list-requested-service-quota-change-history \
        --service-code ec2 --region "$REGION" \
        --query "RequestedQuotas[?QuotaCode=='$SPOT_CODE'].Status" \
        --output text 2>/dev/null | head -1)

    echo "$(date '+%H:%M:%S')  All G and VT Spot: ${SPOT_STATUS:-unknown}"

    if [[ "$SPOT_STATUS" == "APPROVED" ]]; then
        echo "Quota approved! Running terraform apply..."
        terraform apply -auto-approve
        echo "Done. Use 'terraform output ssh_command' to get the SSH command."
        break
    elif [[ "$SPOT_STATUS" == "DENIED" ]]; then
        echo "Spot quota request was denied. Trying on-demand instead..."
        sed -i '' 's/use_spot.*=.*/use_spot = false/' terraform.tfvars
        echo "use_spot set to false. Re-check on-demand quota and run: terraform apply"
        break
    fi

    echo "  Still pending — next check in ${POLL_INTERVAL}s"
    sleep $POLL_INTERVAL
done
