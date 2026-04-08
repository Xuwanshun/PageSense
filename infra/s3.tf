# ── infra/s3.tf ───────────────────────────────────────────────────────────────
# S3 bucket for persistent artifact storage.
#
# WHY WE NEED S3
# ──────────────
# ECS Fargate containers have ephemeral local storage. When a container
# stops (due to deploy, crash, or scale-to-zero), its filesystem is wiped.
# This means processed documents and the vector store would be lost.
#
# S3 is the solution: it is durable (99.999999999% durability), cheap
# ($0.023/GB/month), and works seamlessly with ECS via IAM role permissions.
#
# On container startup: artifacts are synced FROM S3 to local disk.
# After preprocessing/indexing: artifacts are synced TO S3.
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "artifacts" {
  # Bucket names must be globally unique across all AWS accounts.
  # We add the account ID to guarantee uniqueness.
  bucket = "${var.app_name}-artifacts-${data.aws_caller_identity.current.account_id}-${var.environment}"
}

# Block all public access. This bucket should NEVER be publicly readable.
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning: keeps previous versions of files for 30 days.
# Useful for recovering from accidental overwrites of store.json.
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle rule: automatically delete old object versions after 30 days.
# Without this, old versions accumulate and increase storage costs.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}
