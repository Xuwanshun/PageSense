import aws_cdk as cdk

from stacks.network_stack import NetworkStack
from stacks.database_stack import DatabaseStack
from stacks.app_stack import AppStack

app = cdk.App()

# Resolve account from live AWS credentials, region from context or default.
env = cdk.Environment(
    account=app.node.try_get_context("account") or cdk.Aws.ACCOUNT_ID,
    region=app.node.try_get_context("region") or "ca-central-1",
)

# ── Stack 1: Network ──────────────────────────────────────────────────────────
# VPC, subnets, Internet Gateway.
# Rarely changes after initial deploy.
network = NetworkStack(app, "RagAgentNetwork", env=env)

# ── Stack 2: Database ─────────────────────────────────────────────────────────
# RDS PostgreSQL. Stateful — has termination_protection=True.
# Deployed separately so a broken app deploy can never touch the DB.
database = DatabaseStack(app, "RagAgentDatabase", vpc=network.vpc, env=env)
database.add_dependency(network)

# ── Stack 3: Application ──────────────────────────────────────────────────────
# ECS Fargate, ALB, ECR, S3, Secrets, Auto Scaling.
# This is the stack you deploy on every code change.
application = AppStack(
    app, "RagAgentApp",
    vpc=network.vpc,
    db_instance=database.instance,
    env=env,
)
application.add_dependency(database)

app.synth()
