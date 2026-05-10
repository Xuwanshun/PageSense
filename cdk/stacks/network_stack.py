import aws_cdk as cdk
from aws_cdk import aws_ec2 as ec2
from constructs import Construct


class NetworkStack(cdk.Stack):
    """
    Creates the VPC — your private network inside AWS.

    WHAT IS A VPC?
    A VPC (Virtual Private Cloud) is like having your own section of
    AWS's data centre. Resources inside it are isolated from other
    AWS customers by default.

    SUBNET TYPES (two here):
    ┌─────────────────────────────────────────────────────┐
    │                      VPC                           │
    │  ┌──────────────────┐  ┌──────────────────┐        │
    │  │  Public Subnet   │  │  Public Subnet   │        │
    │  │  (AZ-a)          │  │  (AZ-b)          │        │
    │  │  ALB + ECS tasks │  │  ALB + ECS tasks │        │
    │  └──────────────────┘  └──────────────────┘        │
    │  ┌──────────────────┐  ┌──────────────────┐        │
    │  │ Isolated Subnet  │  │ Isolated Subnet  │        │
    │  │  (AZ-a)          │  │  (AZ-b)          │        │
    │  │  RDS database    │  │  RDS database    │        │
    │  └──────────────────┘  └──────────────────┘        │
    └─────────────────────────────────────────────────────┘

    PUBLIC subnet  → has a route to the Internet Gateway → can reach the internet
    ISOLATED subnet → no internet route at all → only reachable from inside the VPC
    """

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        self.vpc = ec2.Vpc(
            self, "Vpc",
            max_azs=2,  # Spread across 2 Availability Zones for redundancy

            # nat_gateways=0 is the key cost-saving decision.
            # NAT Gateways allow private-subnet resources to call OUT to the
            # internet. They cost ~$36/month each. We avoid them by placing
            # ECS tasks in public subnets with assign_public_ip=True instead.
            # The tasks are still protected — Security Groups only allow
            # inbound from the ALB, not from the open internet.
            nat_gateways=0,

            subnet_configuration=[
                # PUBLIC: has a route to the Internet Gateway.
                # The ALB sits here (it must be internet-facing).
                # ECS tasks also sit here (to reach internet without NAT Gateway).
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,  # /24 = 256 addresses per subnet
                ),
                # ISOLATED: completely air-gapped from the internet.
                # Only resources inside the VPC can reach it.
                # RDS lives here — a database should NEVER be internet-accessible.
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
            ],
        )

        # VPC Flow Logs record every network connection (source IP, dest IP,
        # port, bytes, accepted/rejected). Stored in CloudWatch.
        # Essential for debugging "why can't my container reach X?" and
        # for security audits ("who connected to our DB?").
        self.vpc.add_flow_log("FlowLog")

        # Outputs: printed after `cdk deploy` so you can see what was created.
        cdk.CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
        cdk.CfnOutput(
            self, "PublicSubnets",
            value=", ".join([s.subnet_id for s in self.vpc.public_subnets]),
        )
