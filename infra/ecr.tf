# ── infra/ecr.tf ──────────────────────────────────────────────────────────────
# Amazon Elastic Container Registry (ECR) — your private Docker image repository.
#
# WHAT IS ECR?
# ────────────
# ECR is AWS's version of Docker Hub. You push your Docker image to ECR,
# and ECS Fargate pulls it from there when starting your container.
# Using ECR instead of Docker Hub means:
#   - Your image is in the same AWS region as ECS (faster pulls)
#   - No Docker Hub rate limits
#   - Private by default (only your AWS account can pull it)
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_ecr_repository" "app" {
  name                 = "${var.app_name}-${var.environment}"
  image_tag_mutability = "MUTABLE" # allows overwriting the :latest tag

  # Scan images for known vulnerabilities automatically on push.
  # Results appear in the ECR console under "Image scan findings".
  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lifecycle policy: automatically delete old images to save storage costs.
# Without this, ECR accumulates every image you ever pushed (~$0.10/GB/month).
# This policy keeps the 5 most recent images and deletes the rest.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = { type = "expire" }
      }
    ]
  })
}
