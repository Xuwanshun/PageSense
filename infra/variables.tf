# ── infra/variables.tf ────────────────────────────────────────────────────────
# Input variables — things you pass in that differ between environments.
#
# To override a variable from the command line:
#   terraform apply -var="environment=production"
#
# Or create a terraform.tfvars file (not committed to git):
#   environment = "production"
#   aws_region  = "us-west-2"
# ─────────────────────────────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment name (e.g. dev, staging, production)."
  type        = string
  default     = "dev"
}

variable "app_name" {
  description = "Application name used as a prefix for all AWS resource names."
  type        = string
  default     = "rag-agent"
}

variable "container_cpu" {
  description = "ECS task CPU units (1024 = 1 vCPU). Minimum for PaddlePaddle: 2048."
  type        = number
  default     = 2048
}

variable "container_memory" {
  description = "ECS task memory in MB. Minimum for PaddlePaddle: 8192 (8 GB)."
  type        = number
  default     = 8192
}

variable "desired_count" {
  description = "Number of ECS task instances to run. 0 = scale to zero (no cost when idle)."
  type        = number
  default     = 0
}

variable "container_image" {
  description = "Full Docker image URI to deploy (e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest). Updated by CI/CD."
  type        = string
  default     = ""
}
