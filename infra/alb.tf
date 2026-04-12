# ── infra/alb.tf ──────────────────────────────────────────────────────────────
# Application Load Balancer (ALB).
#
# WHAT IS AN ALB?
# ───────────────
# An ALB sits in front of your ECS tasks and handles:
#   - Accepting HTTPS requests from the internet
#   - Forwarding them to healthy ECS containers (on port 8000)
#   - Health checking containers (using GET /health)
#   - TLS termination (decrypts HTTPS so your app only sees HTTP)
#
# REQUEST FLOW:
#   Browser → ALB (HTTPS:443) → ECS Task (HTTP:8000) → Your FastAPI app
#
# NOTE ON HTTPS:
# For HTTPS you need an ACM certificate and a domain name.
# This config creates an HTTP-only ALB for simplicity.
# To add HTTPS:
#   1. Register a domain in Route 53 (or elsewhere)
#   2. Request a certificate in ACM
#   3. Add an aws_lb_listener for port 443 with the certificate ARN
# ─────────────────────────────────────────────────────────────────────────────

# Security group for the ALB — allows inbound HTTP (and optionally HTTPS) from anywhere.
resource "aws_security_group" "alb" {
  name        = "${var.app_name}-alb-${var.environment}"
  description = "Allow HTTP traffic to the Application Load Balancer"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP from internet"
  }

  # Uncomment to also allow HTTPS once you have a certificate:
  # ingress {
  #   from_port   = 443
  #   to_port     = 443
  #   protocol    = "tcp"
  #   cidr_blocks = ["0.0.0.0/0"]
  # }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }
}

# Security group for ECS tasks — only allows traffic FROM the ALB.
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.app_name}-ecs-tasks-${var.environment}"
  description = "Allow traffic from ALB to ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "FastAPI from ALB only"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow outbound (needed for OpenAI API + S3 + ECR)"
  }
}

# The Application Load Balancer itself.
resource "aws_lb" "app" {
  name               = "${var.app_name}-${var.environment}"
  internal           = false  # internet-facing (not private)
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids

  # PDF preprocessing (OCR + layout detection) takes 2-5 minutes per document.
  # Default 60s would timeout every request. 600s gives headroom for large PDFs.
  idle_timeout = 600
}

# Target group: the set of ECS tasks the ALB forwards traffic to.
# The ALB uses /health to determine if a task is healthy before sending it traffic.
resource "aws_lb_target_group" "app" {
  name        = "${var.app_name}-${var.environment}"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"  # required for Fargate (tasks have IPs, not EC2 instance IDs)

  health_check {
    path                = "/health"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 10
    interval            = 30
    # start_period equivalent: the ALB won't mark a task unhealthy for the
    # first 120 seconds (give Paddle time to initialize)
    matcher = "200"
  }

  # Drain connections gracefully before deregistering a task during deploys.
  deregistration_delay = 30
}

# HTTP listener on port 80 — forwards all traffic to the target group.
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
