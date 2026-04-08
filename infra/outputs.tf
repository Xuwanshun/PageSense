# ── infra/outputs.tf ──────────────────────────────────────────────────────────
# Output values — printed after `terraform apply` and readable by CI/CD.
#
# After running terraform apply, copy these values into your GitHub
# repository secrets (Settings → Secrets and variables → Actions).
# ─────────────────────────────────────────────────────────────────────────────

output "ecr_repository_uri" {
  description = "ECR repository URI. Set this as GitHub secret ECR_REPOSITORY_URI."
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name. Set this as GitHub secret ECS_CLUSTER_NAME."
  value       = aws_ecs_cluster.app.name
}

output "ecs_service_name" {
  description = "ECS service name. Set this as GitHub secret ECS_SERVICE_NAME."
  value       = aws_ecs_service.app.name
}

output "ecs_task_definition_family" {
  description = "ECS task definition family. Set this as GitHub secret ECS_TASK_DEFINITION_FAMILY."
  value       = aws_ecs_task_definition.app.family
}

output "alb_dns_name" {
  description = "ALB DNS name. Use this to access your app: http://{alb_dns_name}/health"
  value       = "http://${aws_lb.app.dns_name}"
}

output "s3_bucket_name" {
  description = "S3 bucket for artifacts. Set as S3_BUCKET_NAME in ECS task environment."
  value       = aws_s3_bucket.artifacts.bucket
}
