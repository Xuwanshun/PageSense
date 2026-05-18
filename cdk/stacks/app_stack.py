import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_iam as iam,
    aws_logs as logs,
    aws_rds as rds,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class AppStack(cdk.Stack):
    """
    The application layer — everything that changes when you push code.

    WHAT GETS CREATED:
    - S3 bucket for PDF artifacts and vector store
    - ECR repository (private Docker registry)
    - ECS cluster + Fargate task definition
    - Application Load Balancer (public-facing)
    - ECS Fargate service (desired_count=0 by default = free when idle)
    - Auto Scaling policy (scale on CPU)
    - IAM roles with least-privilege permissions
    - CloudWatch log group

    HOW TO RUN:
      cdk deploy RagAgentApp -c desired_count=2   # start containers
      cdk deploy RagAgentApp -c desired_count=0   # stop (Fargate = $0)

    Or use the helper scripts:
      ./scripts/up.sh    # scale to 2
      ./scripts/down.sh  # scale to 0
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        vpc: ec2.Vpc,
        db_instance: rds.DatabaseInstance,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # How many containers to run. Read from CDK context so you can
        # change it at deploy time without editing code.
        # Default 0 = no containers running = $0 Fargate cost.
        desired_count = int(self.node.try_get_context("desired_count") or 0)
        # ApplicationLoadBalancedFargateService requires >= 1 at creation time.
        # After first deploy, use scripts/down.sh to scale back to 0.
        create_count = max(1, desired_count)

        # ── S3 Bucket ──────────────────────────────────────────────────────────
        # Fargate containers are stateless — their local disk is wiped on every
        # deploy, restart, or scale event. S3 is where we persist:
        #   - Processed PDF artifacts (document.json, chunks.json, images)
        #   - The vector store (store.json)
        # On startup: the app syncs FROM S3 to local disk.
        # After processing: the app syncs TO S3.
        artifacts_bucket = s3.Bucket(
            self, "ArtifactsBucket",
            # S3-managed encryption — data at rest is encrypted automatically.
            encryption=s3.BucketEncryption.S3_MANAGED,
            # Block all public access. This bucket should NEVER be public.
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            # Reject any HTTP (unencrypted) requests — HTTPS only.
            enforce_ssl=True,
            # Versioning keeps previous copies of overwritten files for 30 days.
            # Protects against accidental data loss (e.g., store.json overwrite).
            versioned=True,
            removal_policy=cdk.RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    noncurrent_version_expiration=cdk.Duration.days(30),
                )
            ],
        )

        # ── ECR Repository ─────────────────────────────────────────────────────
        # ECR = Elastic Container Registry = AWS's private Docker Hub.
        # Your CI/CD pipeline builds your Docker image and pushes it here.
        # ECS pulls from here every time it starts a new container.
        # Only keep 5 images — without this, old images accumulate and cost money.
        repository = ecr.Repository(
            self, "Repository",
            removal_policy=cdk.RemovalPolicy.RETAIN,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    max_image_count=5,
                    description="Keep only 5 most recent images",
                )
            ],
        )

        # ── IAM Roles ──────────────────────────────────────────────────────────
        # ECS requires two separate roles. This is confusing at first:
        #
        # EXECUTION ROLE — used by ECS itself (not your app) to:
        #   - Pull the Docker image from ECR
        #   - Fetch secrets from Secrets Manager at startup
        #   - Write container logs to CloudWatch
        #
        # TASK ROLE — used by YOUR APP CODE running inside the container to:
        #   - Read/write S3 objects
        #   - Call any other AWS services your app needs
        #
        # Think of it like: the execution role is the "moving crew" that sets up
        # your apartment. The task role is you, living inside it.

        execution_role = iam.Role(
            self, "ExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                # AWS-managed policy covering ECR pull + CloudWatch logs.
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                ),
            ],
        )

        task_role = iam.Role(
            self, "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # Grant the app read/write access to the artifacts bucket.
        # CDK's grant_read_write() creates the minimal IAM policy needed —
        # just s3:GetObject, s3:PutObject, s3:DeleteObject, s3:ListBucket.
        artifacts_bucket.grant_read_write(task_role)

        # ── Secrets ────────────────────────────────────────────────────────────
        # These three secrets must exist in Secrets Manager BEFORE you deploy.
        # Run scripts/create-secrets.sh to create them.
        #
        # ECS fetches these at container startup and injects them as env vars.
        # Your Python code reads them as regular os.environ — zero AWS SDK calls.
        # The values are never stored in your Docker image or source code.

        # Use full ARNs (including the random suffix AWS appends) so ECS can
        # resolve the secret correctly. from_secret_name_v2 produces incomplete
        # ARNs that cause ResourceNotFoundException at task startup.
        openai_secret = secretsmanager.Secret.from_secret_complete_arn(
            self, "OpenAISecret",
            "arn:aws:secretsmanager:ca-central-1:604561274097:secret:rag-agent/openai-api-key-7KOsxs"
        )
        jwt_secret = secretsmanager.Secret.from_secret_complete_arn(
            self, "JwtSecret",
            "arn:aws:secretsmanager:ca-central-1:604561274097:secret:rag-agent/jwt-secret-key-EMjZaS"
        )
        database_url_secret = secretsmanager.Secret.from_secret_complete_arn(
            self, "DatabaseUrlSecret",
            "arn:aws:secretsmanager:ca-central-1:604561274097:secret:rag-agent/database-url-Q4fgmt"
        )
        google_client_id_secret = secretsmanager.Secret.from_secret_complete_arn(
            self, "GoogleClientIdSecret",
            "arn:aws:secretsmanager:ca-central-1:604561274097:secret:rag-agent/google-client-id-Mo1LtE"
        )
        google_client_secret_secret = secretsmanager.Secret.from_secret_complete_arn(
            self, "GoogleClientSecretSecret",
            "arn:aws:secretsmanager:ca-central-1:604561274097:secret:rag-agent/google-client-secret-ex0dlq"
        )

        # Optional — only referenced when self-hosted VLM is configured.
        # If these secrets don't exist yet, comment them out and the pipeline
        # falls back to gpt-4o automatically.
        vlm_hf_token_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "VlmHfTokenSecret", "rag-agent/vlm-hf-token"
        )
        vlm_base_url_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "VlmBaseUrlSecret", "rag-agent/vlm-base-url"
        )

        # Grant the execution role access to all rag-agent/* secrets using a
        # wildcard ARN. Secrets Manager appends a random suffix to ARNs
        # (e.g. rag-agent/foo-Ab1Cd2), so name-based lookups produce incomplete
        # ARNs that don't match the actual resource in IAM policy evaluation.
        execution_role.add_to_policy(iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:rag-agent/*"],
        ))

        # ── CloudWatch Log Group ───────────────────────────────────────────────
        # All container stdout/stderr goes here.
        # With LOG_FORMAT=json, CloudWatch Logs Insights can query fields like:
        #   fields @timestamp, level, message | filter level = "ERROR"
        # 7-day retention saves cost vs the default (infinite).
        log_group = logs.LogGroup(
            self, "LogGroup",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # ── Task Definition ────────────────────────────────────────────────────
        # The "recipe" ECS follows every time it starts a container.
        # Defines: image, CPU, memory, env vars, secrets, logging, health check.
        task_definition = ecs.FargateTaskDefinition(
            self, "TaskDef",
            cpu=2048,               # 2 vCPU — PaddlePaddle minimum requirement
            memory_limit_mib=8192,  # 8 GB — PaddlePaddle minimum requirement
            execution_role=execution_role,
            task_role=task_role,
            # x86_64 required — PaddlePaddle has a known segfault bug on ARM64
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )

        task_definition.add_container(
            "app",
            image=ecs.ContainerImage.from_ecr_repository(repository, tag="latest"),

            # Non-sensitive config goes here as plain env vars.
            # Sensitive values (API keys, DB passwords) go in `secrets` below.
            environment={
                "APP_MODE":                       "api",
                "LOG_FORMAT":                     "json",
                "LOG_LEVEL":                      "INFO",
                "RAW_DOCUMENTS_DIR":              "/app/data/raw",
                "PROCESSED_DOCUMENTS_DIR":        "/app/data/processed",
                "VECTORSTORE_DIR":                "/app/data/embedded",
                "PADDLE_CACHE_DIR":               "/app/paddle_models",
                "PADDLE_PDX_CACHE_HOME":          "/app/paddle_models",
                "FLAGS_use_mkldnn":               "0",
                "FLAGS_enable_pir_in_executor":   "0",
                "AWS_REGION":                     self.region,
                "S3_BUCKET_NAME":                 artifacts_bucket.bucket_name,
                "USE_DOCUMENT_INTELLIGENCE":      "true",
                "USE_ADAPTIVE_CHUNKING":          "true",
                "USE_VLM_SUMMARIES":              "true",
                "USE_QUERY_ENHANCEMENT":          "true",
                "USE_HYBRID_RETRIEVAL":           "true",
                "USE_LLM_RERANKER":               "true",
                "USE_CONTEXT_COMPRESSION":        "true",
                "USE_FAITHFULNESS_CHECK":         "true",
                "DOC_FILTER_THRESHOLD":           "0.20",
                "HTTPS_ONLY":                     "false",
            },

            # Secrets: ECS fetches these from Secrets Manager at container
            # startup and injects them as environment variables.
            # The container sees them as normal env vars — no code changes needed.
            secrets={
                "OPENAI_API_KEY":        ecs.Secret.from_secrets_manager(openai_secret),
                "JWT_SECRET_KEY":        ecs.Secret.from_secrets_manager(jwt_secret),
                "DATABASE_URL":          ecs.Secret.from_secrets_manager(database_url_secret),
                "GOOGLE_CLIENT_ID":      ecs.Secret.from_secrets_manager(google_client_id_secret),
                "GOOGLE_CLIENT_SECRET":  ecs.Secret.from_secrets_manager(google_client_secret_secret),
                "VLM_HF_TOKEN":          ecs.Secret.from_secrets_manager(vlm_hf_token_secret),
                "VLM_BASE_URL":          ecs.Secret.from_secrets_manager(vlm_base_url_secret),
            },

            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="ecs",
                log_group=log_group,
            ),

            # ECS runs this check every 30s. If it fails 3 times in a row,
            # ECS replaces the container with a fresh one automatically.
            # start_period gives Paddle 120s to load models before checking.
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=cdk.Duration.seconds(30),
                timeout=cdk.Duration.seconds(10),
                retries=3,
                start_period=cdk.Duration.seconds(120),
            ),

            port_mappings=[ecs.PortMapping(container_port=8000)],
        )

        # ── ECS Cluster ────────────────────────────────────────────────────────
        # A logical grouping for your tasks — like a project folder.
        # container_insights=False saves ~$0.15/task/hr while learning.
        # Enable it when you go live (adds CPU/memory graphs in CloudWatch).
        cluster = ecs.Cluster(
            self, "Cluster",
            vpc=vpc,
            container_insights=False,
        )

        # ── ALB + Fargate Service (L3 Pattern) ─────────────────────────────────
        # ApplicationLoadBalancedFargateService is a CDK "L3 construct" —
        # a high-level pattern that creates and wires together:
        #   - Application Load Balancer (internet-facing)
        #   - ALB Listener (port 80)
        #   - Target Group (routes to healthy containers on port 8000)
        #   - ECS Service (keeps desired_count containers running)
        #   - Security Groups for both ALB and ECS tasks
        #
        # This replaces ~100 lines of manual Terraform.
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "Service",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=create_count,

            # Public subnets + assign_public_ip=True is the no-NAT-Gateway pattern.
            # Tasks have public IPs so they can call OpenAI + ECR directly.
            # They're still protected — the auto-created Security Group only
            # allows inbound on port 8000 from the ALB Security Group.
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            assign_public_ip=True,

            public_load_balancer=True,

            # Rolling deployment: bring up new tasks before stopping old ones.
            # min_healthy_percent=50 means ECS can take down half the old tasks
            # while the new version starts. With 2 tasks: one new starts, one old
            # stops, second new starts, second old stops. Zero downtime.
            min_healthy_percent=50,
            max_healthy_percent=200,
        )

        # PDF OCR + layout detection takes 2–5 min per document.
        # The default ALB idle timeout is 60s — it would kill every request.
        # 600s gives headroom for large multi-page PDFs.
        cfn_alb = fargate_service.load_balancer.node.default_child
        cfn_alb.add_property_override(
            "LoadBalancerAttributes",
            [{"Key": "idle_timeout.timeout_seconds", "Value": "600"}],
        )

        # Health check: ALB pings /health every 30s.
        # Only sends traffic to containers returning 200. Sick containers
        # get no traffic and ECS replaces them automatically.
        fargate_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=cdk.Duration.seconds(30),
            timeout=cdk.Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        # RDS port 5432 is opened in DatabaseStack via VPC CIDR to avoid
        # a cross-stack cyclic dependency. No rule needed here.

        # ── Auto Scaling ───────────────────────────────────────────────────────
        # Watches CPU usage. If average CPU across all tasks exceeds 60%,
        # add another container. If it drops, remove containers (down to 0).
        # min_capacity=0 means it can scale all the way to zero when idle.
        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=0,
            max_capacity=4,
        )
        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=60,
            # Cooldown: wait 3 min after scaling before doing it again.
            # Prevents "thrashing" — rapidly adding and removing containers.
            scale_in_cooldown=cdk.Duration.minutes(3),
            scale_out_cooldown=cdk.Duration.minutes(3),
        )

        # ── Outputs ────────────────────────────────────────────────────────────
        cdk.CfnOutput(
            self, "AppUrl",
            value=f"http://{fargate_service.load_balancer.load_balancer_dns_name}",
            description="Your app URL — open this in a browser after scaling up",
        )
        cdk.CfnOutput(
            self, "EcrRepositoryUri",
            value=repository.repository_uri,
            description="Push Docker images here: docker push <this-uri>:latest",
        )
        cdk.CfnOutput(
            self, "S3BucketName",
            value=artifacts_bucket.bucket_name,
            description="S3 bucket for PDF artifacts and vector store",
        )
        cdk.CfnOutput(
            self, "EcsClusterName",
            value=cluster.cluster_name,
            description="Used by scripts/up.sh and scripts/down.sh",
        )
        cdk.CfnOutput(
            self, "EcsServiceName",
            value=fargate_service.service.service_name,
            description="Used by scripts/up.sh and scripts/down.sh",
        )
