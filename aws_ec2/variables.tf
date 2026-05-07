variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 GPU instance type. g4dn.xlarge = T4 16 GB VRAM (~$0.16/hr spot). g5.xlarge = A10G 24 GB VRAM (~$0.34/hr spot)."
  type        = string
  default     = "g4dn.xlarge"
}

variable "use_spot" {
  description = "Use a persistent spot instance (~70 % cheaper than on-demand). The instance is stopped (not terminated) on interruption so EBS data is preserved."
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum spot bid per hour (USD). Set slightly above the current spot price to reduce interruption risk. Check https://aws.amazon.com/ec2/spot/instance-advisor/"
  type        = string
  default     = "0.35"
}

variable "ami_id" {
  description = "AMI for Deep Learning on Ubuntu 22.04 (CUDA pre-installed). Find the latest at https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-x-ubuntu-22-04/"
  type        = string
  # No default — region-specific; must be set in terraform.tfvars
}

variable "key_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "s3_bucket_name" {
  description = "Globally unique S3 bucket name for dataset, code, and checkpoints"
  type        = string
}

variable "hf_token" {
  description = "HuggingFace API token. Required only if the model is gated. Leave empty for public models."
  type        = string
  sensitive   = true
  default     = ""
}

variable "volume_size_gb" {
  description = "Root EBS volume size in GB. 120 GB covers model weights (~10 GB) + dataset + checkpoints + Python env."
  type        = number
  default     = 120
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH. Restrict to your IP (e.g. 1.2.3.4/32) for security."
  type        = string
  default     = "0.0.0.0/0"
}

variable "idle_cpu_threshold" {
  description = "Average CPU utilization (%) below which the instance is considered idle"
  type        = number
  default     = 5
}

variable "idle_evaluation_periods" {
  description = "Number of consecutive 5-minute periods below threshold before auto-stop fires (default 3 = 15 min)"
  type        = number
  default     = 3
}
