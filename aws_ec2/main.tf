terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── S3 ────────────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "training" {
  bucket        = var.s3_bucket_name
  force_destroy = false
  tags          = local.common_tags
}

resource "aws_s3_bucket_versioning" "training" {
  bucket = aws_s3_bucket.training.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "training" {
  bucket                  = aws_s3_bucket.training.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── IAM: EC2 role ─────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ec2_training" {
  name               = "qwen3-vl-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "ec2_policy" {
  # S3: read dataset, write checkpoints (no delete — prevents accidental wipe of cached data)
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.training.arn,
      "${aws_s3_bucket.training.arn}/*",
    ]
  }

  # Self-stop: instance can only stop instances tagged AutoStop=true
  statement {
    actions   = ["ec2:StopInstances"]
    resources = ["arn:aws:ec2:${var.aws_region}:*:instance/*"]
    condition {
      test     = "StringEquals"
      variable = "ec2:ResourceTag/AutoStop"
      values   = ["true"]
    }
  }

  # DescribeInstances: needed for self-stop script to resolve its own instance ID
  statement {
    actions   = ["ec2:DescribeInstances"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "ec2_training" {
  name   = "qwen3-vl-ec2-policy"
  role   = aws_iam_role.ec2_training.id
  policy = data.aws_iam_policy_document.ec2_policy.json
}

resource "aws_iam_instance_profile" "ec2_training" {
  name = "qwen3-vl-ec2-profile"
  role = aws_iam_role.ec2_training.name
}

# ── Security group ────────────────────────────────────────────────────────────

resource "aws_security_group" "training" {
  name        = "qwen3-vl-training-sg"
  description = "SSH access for VLM training instance"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# ── Locals ────────────────────────────────────────────────────────────────────

locals {
  common_tags = {
    Project  = "qwen3-vl-sft"
    AutoStop = "true"
  }

  user_data = base64encode(templatefile("${path.module}/userdata.sh.tpl", {
    s3_bucket  = var.s3_bucket_name
    hf_token   = var.hf_token
    aws_region = var.aws_region
  }))

  # Resolve the actual instance ID regardless of spot vs on-demand
  instance_id = var.use_spot ? (
    length(aws_spot_instance_request.training) > 0
    ? one(aws_spot_instance_request.training[*].spot_instance_id)
    : ""
  ) : (
    length(aws_instance.training) > 0
    ? one(aws_instance.training[*].id)
    : ""
  )
}

# ── EC2: spot instance (persistent — stops instead of terminates on interruption) ──

resource "aws_spot_instance_request" "training" {
  count = var.use_spot ? 1 : 0

  ami           = var.ami_id
  instance_type = var.instance_type

  spot_price                     = var.spot_max_price
  spot_type                      = "persistent"
  instance_interruption_behavior = "stop"
  wait_for_fulfillment           = true

  key_name               = var.key_name
  iam_instance_profile   = aws_iam_instance_profile.ec2_training.name
  vpc_security_group_ids = [aws_security_group.training.id]
  subnet_id              = var.subnet_id != "" ? var.subnet_id : null

  user_data = local.user_data

  root_block_device {
    volume_size           = var.volume_size_gb
    volume_type           = "gp3"
    throughput            = 250
    iops                  = 3000
    encrypted             = true
    delete_on_termination = true
  }

  tags = merge(local.common_tags, { Name = "qwen3-vl-training-spot" })
}

# ── EC2: on-demand instance (fallback when use_spot = false) ──────────────────

resource "aws_instance" "training" {
  count = var.use_spot ? 0 : 1

  ami           = var.ami_id
  instance_type = var.instance_type

  key_name               = var.key_name
  iam_instance_profile   = aws_iam_instance_profile.ec2_training.name
  vpc_security_group_ids = [aws_security_group.training.id]

  user_data = local.user_data

  root_block_device {
    volume_size           = var.volume_size_gb
    volume_type           = "gp3"
    throughput            = 250
    iops                  = 3000
    encrypted             = true
    delete_on_termination = true
  }

  tags = merge(local.common_tags, { Name = "qwen3-vl-training-ondemand" })
}

# ── CloudWatch alarm: stop instance when CPU is idle ─────────────────────────
# Fires after idle_evaluation_periods × 5-minute windows below idle_cpu_threshold.
# Default: CPU < 5 % for 15 consecutive minutes → stop instance.

resource "aws_cloudwatch_metric_alarm" "auto_stop" {
  alarm_name          = "qwen3-vl-idle-auto-stop"
  alarm_description   = "Stop GPU instance when CPU is idle (training finished or never started)"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.idle_evaluation_periods
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = var.idle_cpu_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    InstanceId = local.instance_id
  }

  # Built-in EC2 action — no Lambda required
  alarm_actions = ["arn:aws:automate:${var.aws_region}:ec2:stop"]

  tags = local.common_tags
}
