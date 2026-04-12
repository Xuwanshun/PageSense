# ── infra/ecs.tf ──────────────────────────────────────────────────────────────
# ECS Fargate cluster, task definition, and service.
#
# WHAT IS ECS FARGATE?
# ────────────────────
# ECS (Elastic Container Service) runs Docker containers on AWS.
# Fargate is the "serverless" flavor — AWS manages the underlying
# servers. You just say "run this container with 2 vCPUs and 8 GB RAM"
# and AWS handles the rest.
#
# THREE ECS CONCEPTS:
#   Cluster     — a logical grouping of tasks (like a namespace)
#   Task Definition — the recipe for a container (image, CPU, memory, env vars)
#   Service     — keeps N copies of the task running, restarts on failure,
#                 integrates with the ALB, handles rolling deploys
#
# MENTAL MODEL:
#   Task Definition ≈ docker-compose.yml (describes how to run the container)
#   Service         ≈ the process that ENSURES it keeps running
# ─────────────────────────────────────────────────────────────────────────────

# CloudWatch log group — where container stdout/stderr (your logs) are stored.
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}-${var.environment}"
  retention_in_days = 30  # keep logs for 30 days, then auto-delete
}

# ECS Cluster — a logical namespace for our tasks.
resource "aws_ecs_cluster" "app" {
  name = "${var.app_name}-${var.environment}"

  # Enable Container Insights for CloudWatch metrics (CPU, memory, etc.)
  # Adds ~$0.15/task/hour. Comment out to save cost on a dev environment.
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Task Definition — describes exactly how to run the container.
# This is updated every time CI/CD pushes a new image (with a new image tag).
resource "aws_ecs_task_definition" "app" {
  family                   = "${var.app_name}-${var.environment}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"  # required for Fargate
  cpu                      = var.container_cpu     # 2048 = 2 vCPU
  memory                   = var.container_memory  # 8192 = 8 GB

  execution_role_arn = aws_iam_role.ecs_task_execution.arn  # ECS pulls image + secrets
  task_role_arn      = aws_iam_role.ecs_task.arn            # app accesses S3

  # EFS volume for Paddle model cache (models persist across container restarts)
  volume {
    name = "paddle-models"
    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.paddle_models.id
      transit_encryption      = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.paddle_models.id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name  = "app"
      image = var.container_image != "" ? var.container_image : "${aws_ecr_repository.app.repository_url}:latest"

      portMappings = [{ containerPort = 8000, hostPort = 8000, protocol = "tcp" }]

      # Container environment variables.
      # Non-sensitive values go here. Secrets go in the `secrets` block below.
      environment = [
        { name = "APP_MODE",                value = "api" },
        { name = "LOG_FORMAT",              value = "json" },
        { name = "LOG_LEVEL",               value = "INFO" },
        { name = "RAW_DOCUMENTS_DIR",       value = "/app/data/raw" },
        { name = "PROCESSED_DOCUMENTS_DIR", value = "/app/data/processed" },
        { name = "VECTORSTORE_DIR",         value = "/app/data/embedded" },
        { name = "PADDLE_CACHE_DIR",        value = "/app/paddle_models" },
        { name = "PADDLE_PDX_CACHE_HOME",   value = "/app/paddle_models" },
        { name = "FLAGS_use_mkldnn",             value = "0" },
        { name = "FLAGS_enable_pir_in_executor", value = "0" },
        { name = "AWS_REGION",              value = var.aws_region },
        { name = "S3_BUCKET_NAME",          value = aws_s3_bucket.artifacts.bucket },
      ]

      # Secrets: ECS pulls these from Secrets Manager at container startup
      # and injects them as environment variables. The application code
      # reads them as regular env vars — no AWS SDK calls needed.
      #
      # HOW TO CREATE THE SECRET:
      #   aws secretsmanager create-secret \
      #     --name "rag-agent/openai-api-key" \
      #     --secret-string '{"OPENAI_API_KEY":"sk-your-real-key"}'
      secrets = [
        {
          name      = "OPENAI_API_KEY"
          valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.app_name}/openai-api-key"
        }
      ]

      # Mount the EFS volume for Paddle models
      mountPoints = [
        {
          containerPath = "/app/paddle_models"
          sourceVolume  = "paddle-models"
          readOnly      = false
        }
      ]

      # CloudWatch Logs: where container stdout/stderr goes.
      # Set LOG_FORMAT=json so CloudWatch Insights can query fields.
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.app.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      # Health check using the /health endpoint.
      # ECS replaces the container if /health returns non-200 three times.
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 120  # wait 120s before checking (Paddle init time)
      }
    }
  ])
}

# ECS Service — keeps the task running and handles rolling deploys.
resource "aws_ecs_service" "app" {
  name            = "${var.app_name}-${var.environment}"
  cluster         = aws_ecs_cluster.app.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count  # 0 = scale to zero when not needed
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    # assign_public_ip = true allows the container to reach the internet
    # (needed for OpenAI API and ECR image pull) WITHOUT a NAT Gateway.
    # A NAT Gateway costs ~$32/month — assign_public_ip = true avoids this.
    # Security note: the container is only reachable from the ALB security group.
    assign_public_ip = true
  }

  # Connect the service to the ALB so it receives traffic.
  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8000
  }

  # Rolling deployment settings.
  # With minimum_healthy_percent=50, ECS brings up a new task before
  # stopping the old one — zero-downtime deployments.
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  depends_on = [aws_lb_listener.http, aws_iam_role_policy_attachment.ecs_task_execution_managed]
}
