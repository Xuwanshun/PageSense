# ── infra/main.tf ─────────────────────────────────────────────────────────────
# Terraform root configuration.
#
# WHAT IS TERRAFORM?
# ──────────────────
# Terraform is a tool that lets you describe AWS infrastructure as code.
# Instead of clicking through the AWS console (which is not repeatable),
# you write .tf files and Terraform creates/updates/deletes the resources.
#
# Key commands:
#   terraform init      → download provider plugins
#   terraform plan      → show what WOULD change (safe, no changes made)
#   terraform apply     → create/update infrastructure
#   terraform destroy   → delete everything (careful!)
#
# MENTAL MODEL: Terraform as a "desired state" engine.
# You describe what you WANT. Terraform figures out what to create/modify/delete
# to make AWS match your description.
#
# REMOTE STATE
# ────────────
# Terraform keeps track of what it created in a "state file".
# We store this in S3 (not locally) so:
#   - Multiple developers can share the same state
#   - State is not lost if your laptop breaks
#   - CI/CD can deploy without needing state on the CI runner
#
# SETUP BEFORE FIRST USE:
# 1. Create an S3 bucket for Terraform state:
#    aws s3 mb s3://YOUR-STATE-BUCKET-NAME --region us-east-1
# 2. Update the bucket name in the backend "s3" block below.
# 3. Run: terraform init
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Stores Terraform state in S3 so it is shared and durable.
  # Replace "YOUR-STATE-BUCKET-NAME" with your actual bucket name.
  # You must create this bucket manually (chicken-and-egg: Terraform can't
  # create the bucket that stores its own state).
  backend "s3" {
    bucket = "my-rag-agent-tf-state-604561274097"
    key    = "rag-agent/terraform.tfstate"
    region = "ca-central-1"
    # Enable state locking via DynamoDB (prevents concurrent deploys
    # from corrupting the state file). Optional but recommended.
    # dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region

  # Tag every AWS resource with project and environment info.
  # This makes it easy to find all resources belonging to this project
  # and understand your AWS bill.
  default_tags {
    tags = {
      Project     = "rag-agent"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Data sources ──────────────────────────────────────────────────────────────
# Data sources read existing AWS resources (they don't create anything).

# Get information about the current AWS account and region.
# Used to build ARNs (Amazon Resource Names) in IAM policies.
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Look up the default VPC — we deploy into it for simplicity.
# In a production system you would create a dedicated VPC.
data "aws_vpc" "default" {
  default = true
}

# Look up subnets in the default VPC.
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
