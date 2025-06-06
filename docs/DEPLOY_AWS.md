# Deploying the Crypto Trading Bot to AWS Fargate

This document outlines the steps and configurations required to deploy the `trading-bot-test:latest` Docker image to AWS Fargate with a cost-focused approach.

## 1. ECR Repository Creation

First, you need a repository in Amazon Elastic Container Registry (ECR) to store your Docker image.

**a. Create ECR Repository:**
Use the following AWS CLI command to create an ECR repository named `crypto-trading-bot`. Replace `<YOUR_REGION>` with your desired AWS region (e.g., `us-east-1`).

```bash
aws ecr create-repository --repository-name crypto-trading-bot --region <YOUR_REGION> --image-scanning-configuration scanOnPush=true
```

**b. Tag and Push Local Image to ECR:**
Replace `<YOUR_ACCOUNT_ID>` and `<YOUR_REGION>` in the commands below.

```bash
# Authenticate Docker to your ECR registry
aws ecr get-login-password --region <YOUR_REGION> | sudo docker login --username AWS --password-stdin <YOUR_ACCOUNT_ID>.dkr.ecr.<YOUR_REGION>.amazonaws.com

# Tag your local image
sudo docker tag trading-bot-test:latest <YOUR_ACCOUNT_ID>.dkr.ecr.<YOUR_REGION>.amazonaws.com/crypto-trading-bot:latest

# Push the image to ECR
sudo docker push <YOUR_ACCOUNT_ID>.dkr.ecr.<YOUR_REGION>.amazonaws.com/crypto-trading-bot:latest
```

## 2. Secrets Management Strategy

API keys and other sensitive information must be stored securely using AWS Secrets Manager. The application will be configured to fetch these secrets at runtime.

**a. Store Coinbase API Credentials:**
Create two secrets in AWS Secrets Manager: one for the Coinbase API key and one for the API secret.

*   **API Key Secret:**
    Replace `YOUR_COINBASE_API_KEY_VALUE` with your actual API key.

    ```bash
    aws secretsmanager create-secret --name prod/crypto-bot/coinbase-api-key --description "Coinbase API Key for Crypto Trading Bot" --secret-string "YOUR_COINBASE_API_KEY_VALUE" --region <YOUR_REGION>
    ```

*   **API Secret Secret:**
    Replace `YOUR_COINBASE_API_SECRET_VALUE` with your actual API secret.

    ```bash
    aws secretsmanager create-secret --name prod/crypto-bot/coinbase-api-secret --description "Coinbase API Secret for Crypto Trading Bot" --secret-string "YOUR_COINBASE_API_SECRET_VALUE" --region <YOUR_REGION>
    ```

**b. Store RDS Database Credentials (if using RDS):**
If you are using RDS for the database, store the database credentials (username, password, host, port, dbname) as a JSON string in a secret.

*   **DB Credentials Secret:**
    Replace the placeholder values in the `--secret-string` JSON.

    ```bash
    aws secretsmanager create-secret --name prod/crypto-bot/db-credentials --description "RDS DB Credentials for Crypto Trading Bot" --secret-string '{"username":"yourdbuser","password":"yourdbpassword","host":"yourdbhost.rds.amazonaws.com","port":5432,"dbname":"yourdbname"}' --region <YOUR_REGION>
    ```

**Important:** The application code (specifically `crypto_trading/config.py` and potentially other parts that handle API client initialization) **must be adapted** to:
1.  Read the ARNs of these secrets from environment variables.
2.  Use the AWS SDK (boto3) to fetch the actual secret values from AWS Secrets Manager using these ARNs.
The provided `crypto_trading/config.py` modifications already include logic for fetching DB credentials if `USE_RDS` is true. Similar logic would be needed for Coinbase API keys.

## 3. ECS Task Definition

The Task Definition specifies the Docker container to run, CPU/memory, roles, environment variables, and logging.

Save the following JSON as `task-definition.json`. **You MUST replace all placeholder values** (e.g., `<YOUR_ACCOUNT_ID>`, `<YOUR_REGION>`, `<YOUR_TASK_ROLE_ARN_HERE>`, ARNs for secrets).

```json
{
  "family": "crypto-trading-bot-task",
  "taskRoleArn": "<YOUR_TASK_ROLE_ARN_HERE>",
  "executionRoleArn": "<YOUR_EXECUTION_ROLE_ARN_HERE>",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "crypto-trading-bot-container",
      "image": "<YOUR_ACCOUNT_ID>.dkr.ecr.<YOUR_REGION>.amazonaws.com/crypto-trading-bot:latest",
      "essential": true,
      "command": ["-c", "config/trading_COINBASE.json", "-l", "INFO"],
      "environment": [
        {
          "name": "COINBASE_API_KEY_SECRET_ARN",
          "value": "arn:aws:secretsmanager:<YOUR_REGION>:<YOUR_ACCOUNT_ID>:secret:prod/crypto-bot/coinbase-api-key-XXXXXX"
        },
        {
          "name": "COINBASE_API_SECRET_SECRET_ARN",
          "value": "arn:aws:secretsmanager:<YOUR_REGION>:<YOUR_ACCOUNT_ID>:secret:prod/crypto-bot/coinbase-api-secret-XXXXXX"
        },
        {
          "name": "USE_RDS",
          "value": "true"
        },
        {
          "name": "DB_SECRET_NAME",
          "value": "arn:aws:secretsmanager:<YOUR_REGION>:<YOUR_ACCOUNT_ID>:secret:prod/crypto-bot/db-credentials-XXXXXX"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/crypto-trading-bot-task",
          "awslogs-region": "<YOUR_REGION>",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Permissions for Task Role (`<YOUR_TASK_ROLE_ARN_HERE>`):**
This IAM role needs permissions to access the secrets. Attach a policy like this:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "secretsmanager:GetSecretValue",
            "Resource": [
                "arn:aws:secretsmanager:<YOUR_REGION>:<YOUR_ACCOUNT_ID>:secret:prod/crypto-bot/coinbase-api-key-*",
                "arn:aws:secretsmanager:<YOUR_REGION>:<YOUR_ACCOUNT_ID>:secret:prod/crypto-bot/coinbase-api-secret-*",
                "arn:aws:secretsmanager:<YOUR_REGION>:<YOUR_ACCOUNT_ID>:secret:prod/crypto-bot/db-credentials-*"
            ]
        }
    ]
}
```
The `prod/crypto-bot/db-credentials-*` resource should only be included if using RDS. The `-*` at the end of ARNs is a best practice if the secret has a system-generated suffix. If you used exact names, adjust accordingly.

**ECS Task Execution Role (`<YOUR_EXECUTION_ROLE_ARN_HERE>`):**
This is a standard IAM role that grants ECS permissions to pull images from ECR and write logs to CloudWatch. It typically has the `AmazonECSTaskExecutionRolePolicy` managed policy attached.

**Register the Task Definition:**
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json --region <YOUR_REGION>
```

## 4. ECS Service Definition

The ECS Service maintains the desired number of instances of the Task Definition.

**Create the Service:**
Replace placeholders. Use `FARGATE_SPOT` for potential cost savings, or `FARGATE` for on-demand instances.
Ensure your VPC subnets have a NAT Gateway (if private) or an Internet Gateway (if public) for outbound internet access (e.g., to reach Coinbase API and AWS services).
The security group should allow outbound HTTPS (port 443) traffic.

```bash
aws ecs create-service \
    --cluster default \
    --service-name crypto-trading-bot-service \
    --task-definition crypto-trading-bot-task \
    --desired-count 1 \
    --launch-type FARGATE \
    --capacity-provider-strategy '[{"capacityProvider":"FARGATE_SPOT", "weight":1, "base":0}]' \
    --network-configuration "awsvpcConfiguration={subnets=[\"<YOUR_PRIVATE_SUBNET_ID_1>\",\"<YOUR_PRIVATE_SUBNET_ID_2>\"],securityGroups=[\"<YOUR_SECURITY_GROUP_ID>\"],assignPublicIp=DISABLED}" \
    --region <YOUR_REGION>
```
If `FARGATE_SPOT` is not suitable or available, change to `--capacity-provider-strategy '[{"capacityProvider":"FARGATE", "weight":1, "base":1}]'` or remove the `capacityProviderStrategy` parameter entirely to default to FARGATE on-demand. For `assignPublicIp=ENABLED`, ensure your subnets are public.

This guide provides a foundational setup. Depending on your needs, you might also consider:
- Setting up Auto Scaling for the service.
- Configuring more detailed health checks.
- Integrating with Application Load Balancers if exposing an API.
- Using CloudFormation or CDK for infrastructure as code.
```
