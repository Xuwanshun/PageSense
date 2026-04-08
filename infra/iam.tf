# ── infra/iam.tf ──────────────────────────────────────────────────────────────
# IAM roles and policies for ECS.
#
# TWO ROLES YOU NEED FOR ECS
# ──────────────────────────
# ECS uses two distinct IAM roles. This is confusing at first, but important:
#
# 1. Task Execution Role
#    Used by ECS ITSELF (the control plane) to:
#      - Pull the Docker image from ECR
#      - Read secrets from Secrets Manager
#      - Write logs to CloudWatch
#    Your application code CANNOT use this role.
#
# 2. Task Role
#    Used by YOUR APPLICATION running inside the container to:
#      - Read/write objects in S3
#      - Call AWS APIs
#    The task execution role can NOT be used by your code.
#
# Mental model: the execution role is the "delivery driver" (sets up the
# container), and the task role is the "worker inside" (runs your app).
# ─────────────────────────────────────────────────────────────────────────────

# ── Task Execution Role ───────────────────────────────────────────────────────

resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.app_name}-task-execution-${var.environment}"

  # Trust policy: allows ECS tasks to assume this role.
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Attach the AWS-managed policy that grants standard ECS execution permissions.
# This covers: ECR image pull, CloudWatch log creation, SSM parameter access.
resource "aws_iam_role_policy_attachment" "ecs_task_execution_managed" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Additional policy: allow ECS to read the OPENAI_API_KEY from Secrets Manager.
# The secret must exist at: arn:aws:secretsmanager:REGION:ACCOUNT:secret:rag-agent/openai-api-key
resource "aws_iam_role_policy" "ecs_task_execution_secrets" {
  name = "read-openai-secret"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret",
      ]
      Resource = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.app_name}/*"
    }]
  })
}

# ── Task Role (used by application code) ─────────────────────────────────────

resource "aws_iam_role" "ecs_task" {
  name = "${var.app_name}-task-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Allow the application to read and write objects in the artifacts S3 bucket.
# This is what storage/s3.py uses for syncing processed artifacts.
resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "s3-artifacts-access"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*",
        ]
      }
    ]
  })
}
