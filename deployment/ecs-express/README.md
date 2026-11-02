# ECS Express Mode deployment

This folder prepares the API for an Amazon ECS Express Mode service. The
deployment script is deliberately not called by CI and has not been run. A
service deployment creates billable AWS resources.

## Before creating a service

1. Review the expected spend for the selected AWS Region. ECS Express Mode has
   no additional service fee, but the resources it creates are billed: at
   least one Fargate task, an Application Load Balancer, CloudWatch logs and
   metrics, and data transfer. The script caps autoscaling at one task and
   keeps Express Mode's default 1 vCPU / 2 GiB task size; that task still runs
   continuously. Private subnet egress may also require NAT or VPC endpoints.
   Review an account- and traffic-specific estimate before creating the
   service; no fixed monthly estimate is included here.
2. Choose public or private ingress. The script defaults to private and
   requires explicit subnets and security groups. Private subnets result in an
   internal load balancer. Public ingress is blocked unless
   `-ConfirmPublicIngress` is supplied. The API currently has no authentication,
   so do not make it internet-facing until access control is added.
3. Prepare the IAM roles. The task execution role needs the
   `AmazonECSTaskExecutionRolePolicy` for ECR pulls and CloudWatch logging. The
   infrastructure role needs
   `AmazonECSInfrastructureRoleforExpressGatewayServices`. The task role must
   have only `s3:GetObject` for the configured model and history objects:
   `models/xgboost_final_model.joblib` and `serving/history.csv` in the chosen
   bucket. Do not put AWS credentials in the container environment.
4. Ensure the selected image tag is the commit SHA tag published by the CI
   workflow, not the mutable `latest` tag.

The task role's S3 policy should be scoped to those two object ARNs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadForecastArtifacts",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::YOUR_BUCKET/models/xgboost_final_model.joblib",
        "arn:aws:s3:::YOUR_BUCKET/serving/history.csv"
      ]
    }
  ]
}
```

After creation, set a finite retention period on the service's CloudWatch log
group. ECS Express Mode log groups do not expire by default and remain after
the service is deleted.

## Preview service creation

Use at least two subnets in separate Availability Zones. For private ingress,
use private subnets and make sure the tasks can reach ECR, S3, and CloudWatch
Logs through the network path you selected.

```powershell
$network = @{
  subnets = @("subnet-private-a", "subnet-private-b")
  securityGroups = @("sg-smartdemand")
} | ConvertTo-Json -Compress

./deployment/ecs-express/deploy.ps1 `
  -Action Create `
  -ImageUri "123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/smartdemand-api:<commit-sha>" `
  -S3Bucket "your-model-artifact-bucket" `
  -TaskRoleArn "arn:aws:iam::123456789012:role/smartdemand-task" `
  -TaskExecutionRoleArn "arn:aws:iam::123456789012:role/ecsTaskExecutionRole" `
  -InfrastructureRoleArn "arn:aws:iam::123456789012:role/ecsInfrastructureRoleForExpressServices" `
  -NetworkConfigurationJson $network `
  -Ingress Private `
  -WhatIf
```

`-WhatIf` previews the intended operation without calling AWS. Remove it only
after reviewing the network, role policies, and estimated spend. PowerShell
also prompts before the high-impact create or update operation.

## Update an existing service

Pass the service ARN returned by ECS and the SHA-tagged image to deploy a new
revision. The script keeps the current network configuration for updates.

```powershell
./deployment/ecs-express/deploy.ps1 `
  -Action Update `
  -ServiceArn "arn:aws:ecs:ap-southeast-2:123456789012:service/default/smartdemand-api" `
  -ImageUri "123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/smartdemand-api:<commit-sha>" `
  -S3Bucket "your-model-artifact-bucket" `
  -TaskRoleArn "arn:aws:iam::123456789012:role/smartdemand-task" `
  -WhatIf
```

The local Prometheus/Grafana stack remains for local development. ECS Express
Mode also provisions CloudWatch logging and monitoring for the cloud service;
the local Prometheus target does not scrape the cloud service automatically.
