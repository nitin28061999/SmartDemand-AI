[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param(
    [ValidateSet('Create', 'Update')]
    [string]$Action = 'Create',

    [Parameter(Mandatory = $true)]
    [string]$ImageUri,

    [Parameter(Mandatory = $true)]
    [string]$S3Bucket,

    [Parameter(Mandatory = $true)]
    [string]$TaskRoleArn,

    [string]$TaskExecutionRoleArn,
    [string]$InfrastructureRoleArn,
    [string]$ServiceArn,
    [string]$ServiceName = 'smartdemand-api',
    [string]$NetworkConfigurationJson,

    [ValidateSet('Private', 'Public')]
    [string]$Ingress = 'Private',

    [switch]$ConfirmPublicIngress,

    [string]$Region = 'ap-southeast-2'
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw 'AWS CLI v2 is required. Install and configure it before deploying.'
}

if ($ImageUri -match ':latest$') {
    throw 'Use the immutable commit SHA image tag instead of :latest.'
}

if ($Ingress -eq 'Public' -and -not $ConfirmPublicIngress) {
    throw 'Public ingress needs -ConfirmPublicIngress. The API currently has no authentication.'
}

$containerConfig = @{
    image = $ImageUri
    containerPort = 8000
    environment = @(
        @{ name = 'S3_BUCKET'; value = $S3Bucket },
        @{ name = 'AWS_REGION'; value = $Region }
    )
} | ConvertTo-Json -Depth 5 -Compress

$scalingTarget = @{
    minTaskCount = 1
    maxTaskCount = 1
    autoScalingMetric = 'AVERAGE_CPU'
    autoScalingTargetValue = 60
} | ConvertTo-Json -Compress

if ($Action -eq 'Create') {
    if (-not $TaskExecutionRoleArn) {
        throw 'TaskExecutionRoleArn is required when creating the service.'
    }
    if (-not $InfrastructureRoleArn) {
        throw 'InfrastructureRoleArn is required when creating the service.'
    }
    if (-not $NetworkConfigurationJson) {
        throw 'NetworkConfigurationJson is required. Choose subnets and security groups explicitly.'
    }

    try {
        $networkConfiguration = $NetworkConfigurationJson | ConvertFrom-Json
    }
    catch {
        throw "NetworkConfigurationJson is invalid JSON: $($_.Exception.Message)"
    }

    if (@($networkConfiguration.subnets).Count -lt 2) {
        throw 'Provide at least two subnets in NetworkConfigurationJson.'
    }
    if (@($networkConfiguration.securityGroups).Count -lt 1) {
        throw 'Provide at least one security group in NetworkConfigurationJson.'
    }

    if ($Ingress -eq 'Private') {
        Write-Host 'Private ingress selected. Confirm the supplied subnets have no direct route to an Internet Gateway.'
    }
    else {
        Write-Warning 'This creates an internet-facing API without application authentication.'
    }

    $awsArgs = @(
        'ecs', 'create-express-gateway-service',
        '--region', $Region,
        '--service-name', $ServiceName,
        '--execution-role-arn', $TaskExecutionRoleArn,
        '--infrastructure-role-arn', $InfrastructureRoleArn,
        '--task-role-arn', $TaskRoleArn,
        '--network-configuration', $NetworkConfigurationJson,
        '--primary-container', $containerConfig,
        '--health-check-path', '/health',
        '--scaling-target', $scalingTarget,
        '--monitor-resources', 'DEPLOYMENT',
        '--monitor-mode', 'TEXT-ONLY'
    )
    $target = "ECS Express Mode service '$ServiceName' in $Region"
}
else {
    if (-not $ServiceArn) {
        throw 'ServiceArn is required when updating the service.'
    }

    $awsArgs = @(
        'ecs', 'update-express-gateway-service',
        '--region', $Region,
        '--service-arn', $ServiceArn,
        '--task-role-arn', $TaskRoleArn,
        '--primary-container', $containerConfig,
        '--health-check-path', '/health',
        '--scaling-target', $scalingTarget,
        '--monitor-resources', 'DEPLOYMENT',
        '--monitor-mode', 'TEXT-ONLY'
    )
    $target = "ECS Express Mode service '$ServiceArn' in $Region"
}

if ($PSCmdlet.ShouldProcess($target, "$Action service with image '$ImageUri'")) {
    & aws @awsArgs
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI failed with exit code $LASTEXITCODE."
    }
}
