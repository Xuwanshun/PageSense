# ── infra/efs.tf ──────────────────────────────────────────────────────────────
# Amazon Elastic File System (EFS) — shared network filesystem for Paddle models.
#
# WHY WE NEED EFS
# ───────────────
# PaddleOCR models are ~1.5 GB. We bake them into the Docker image in Stage 3,
# but EFS provides a faster alternative for production:
#
# WITHOUT EFS: Models are in the image. Cold start is fast but the image is huge.
# WITH EFS:    Models are on EFS, mounted at /app/paddle_models when the container
#              starts. The first container to start downloads models to EFS.
#              All subsequent containers (new deploys, scale events) find models
#              already on EFS and skip the download entirely.
#
# EFS cost: ~$0.30/GB/month. For ~2 GB of models: ~$0.60/month.
#
# For this project (models baked into image), EFS is optional. It is here
# for when you want to separate model management from application code.
# To enable: set the EFS mount in the ECS task definition (ecs.tf).
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_efs_file_system" "paddle_models" {
  # Bursting throughput is sufficient for model loading (sequential reads).
  # No ongoing writes happen to this filesystem after models are downloaded.
  throughput_mode = "bursting"

  # Standard storage class (not Infrequent Access) since models are
  # read every time a new container starts.
  lifecycle_policy {
    transition_to_ia = "AFTER_90_DAYS"
  }

  tags = {
    Name = "${var.app_name}-paddle-models-${var.environment}"
  }
}

# Security group for EFS — allows NFS traffic (port 2049) from ECS tasks.
resource "aws_security_group" "efs" {
  name        = "${var.app_name}-efs-${var.environment}"
  description = "Allow NFS access to EFS from ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }
}

# Mount target: one per subnet (allows ECS tasks in that subnet to access EFS).
# We create one for each default VPC subnet.
resource "aws_efs_mount_target" "paddle_models" {
  for_each = toset(data.aws_subnets.default.ids)

  file_system_id  = aws_efs_file_system.paddle_models.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

# Access point: a specific entry point into the EFS filesystem.
# Setting uid/gid to 1001 matches the appuser in the Dockerfile.
resource "aws_efs_access_point" "paddle_models" {
  file_system_id = aws_efs_file_system.paddle_models.id

  posix_user {
    uid = 1001
    gid = 1001
  }

  root_directory {
    path = "/paddle_models"
    creation_info {
      owner_uid   = 1001
      owner_gid   = 1001
      permissions = "755"
    }
  }
}
