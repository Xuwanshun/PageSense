output "instance_id" {
  description = "EC2 instance ID"
  value = var.use_spot ? (
    length(aws_spot_instance_request.training) > 0
    ? one(aws_spot_instance_request.training[*].spot_instance_id)
    : null
  ) : (
    length(aws_instance.training) > 0
    ? one(aws_instance.training[*].id)
    : null
  )
}

output "public_ip" {
  description = "Public IP address for SSH access"
  value = var.use_spot ? (
    length(aws_spot_instance_request.training) > 0
    ? one(aws_spot_instance_request.training[*].public_ip)
    : null
  ) : (
    length(aws_instance.training) > 0
    ? one(aws_instance.training[*].public_ip)
    : null
  )
}

output "s3_bucket" {
  description = "S3 bucket for training data and checkpoints"
  value       = aws_s3_bucket.training.bucket
}

output "ssh_command" {
  description = "SSH command to connect (replace the key path if needed)"
  value = var.use_spot ? (
    length(aws_spot_instance_request.training) > 0
    ? "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${one(aws_spot_instance_request.training[*].public_ip)}"
    : "spot request not yet fulfilled"
  ) : (
    length(aws_instance.training) > 0
    ? "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${one(aws_instance.training[*].public_ip)}"
    : "no instance provisioned"
  )
}

output "upload_code_command" {
  description = "Command to upload your training code to S3 before starting"
  value       = "aws s3 sync . s3://${var.s3_bucket_name}/code/ --exclude '.git/*' --region ${var.aws_region}"
}
