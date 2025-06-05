# CloudWatch Logging and Monitoring for Crypto Trading Bot on AWS Fargate

This document outlines a cost-effective logging and monitoring strategy using AWS CloudWatch for the crypto trading bot deployed on AWS Fargate.

## 1. CloudWatch Logs

Effective log management is crucial for debugging and auditing. Fargate's integration with CloudWatch Logs provides a centralized place for application and system logs.

**a. Fargate Log Configuration:**
As specified in the ECS Task Definition, the Fargate service is configured to use the `awslogs` log driver.
-   **Log Group Name:** `/ecs/crypto-trading-bot-task` (or as configured in your Task Definition).
-   All `stdout` and `stderr` output from the container, which includes Python's `logging` output, will be automatically sent to this log group.

**b. Log Retention (Action Required for Cost Management):**
By default, CloudWatch Logs are stored indefinitely, which can lead to increasing storage costs. It is **highly recommended** to set a log retention period. For example, 30 days might be a good balance between availability for debugging and cost.

**AWS CLI Command to Set Log Retention (example: 30 days):**
Replace `<YOUR_REGION>` with your AWS region.

```bash
aws logs put-retention-policy \
    --log-group-name "/ecs/crypto-trading-bot-task" \
    --retention-in-days 30 \
    --region <YOUR_REGION>
```
This command should be run once per log group.

**c. Python Application Logging:**
The standard Python `logging` module, when configured (e.g., via `logging.basicConfig` or more advanced configurations that output to `stdout`/`stderr`), works seamlessly with Fargate's log collection. No special libraries are needed for basic log shipping. Ensure your application's log level (`INFO`, `DEBUG`, `ERROR`) is appropriately configured for the desired verbosity.

## 2. CloudWatch Metrics

Metrics provide quantitative data about the performance and health of your application and infrastructure.

**a. Standard Fargate Metrics (Free Tier & Default):**
AWS Fargate automatically sends several key metrics to CloudWatch under the `AWS/ECS` namespace (specifically for services running on Fargate). These include:
-   **CPUUtilization:** The percentage of CPU units used by your tasks. Essential for understanding if your tasks are CPU-bound and for optimizing task sizing (cost).
-   **MemoryUtilization:** The percentage of memory used by your tasks. Crucial for monitoring for memory leaks, optimizing task sizing (cost), and preventing OutOfMemory errors.
-   **RunningTaskCount:** (Found under `AWS/ECS` > `ClusterName`, `ServiceName`) The number of tasks currently running for the service.

These metrics are vital for basic monitoring, auto-scaling (if configured), and ensuring your service is running as expected. They are part of the standard CloudWatch free tier or have predictable costs based on usage.

**b. Custom Metrics (Cost Consideration):**
You can publish custom metrics from your application to CloudWatch for more detailed monitoring specific to your bot's operations (e.g., `TradesSuccessful`, `ApiErrors`, `PortfolioValue`).

-   **Cost Implication:** Custom metrics are **chargeable** per metric per month, plus charges for `PutMetricData` API calls. Therefore, they should be used **sparingly** and only for truly essential business or operational insights.
-   **Implementation (if needed):**
    -   The Fargate Task IAM Role would require the `cloudwatch:PutMetricData` permission for the relevant metric namespace.
        ```json
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "cloudwatch:PutMetricData",
                    "Resource": "*" // Or restrict to a specific namespace if preferred
                }
            ]
        }
        ```
    -   The application would use the AWS SDK (Boto3) to call `put_metric_data`.
        ```python
        # Example (conceptual - not for direct implementation now)
        # import boto3
        # cloudwatch = boto3.client('cloudwatch', region_name='<YOUR_REGION>')
        # cloudwatch.put_metric_data(
        #     MetricData=[
        #         {
        #             'MetricName': 'TradesSuccessful',
        #             'Dimensions': [
        #                 {'Name': 'TradingPair', 'Value': 'BTC-USD'},
        #             ],
        #             'Unit': 'Count',
        #             'Value': 1
        #         },
        #     ],
        #     Namespace='CryptoTradingBot/Application'
        # )
        ```
    -   Consider aggregating metrics within the application before publishing (e.g., count of errors per minute rather than for every error) to reduce API calls and costs.

## 3. CloudWatch Alarms (Cost Consideration)

Alarms notify you or take automated actions when a metric breaches a defined threshold.

**a. Recommended Alarms:**
-   **High CPU Utilization:**
    -   Metric: `CPUUtilization` (from `AWS/ECS`)
    -   Threshold: e.g., > 80%
    -   Period: e.g., for 15 consecutive minutes
    -   Action: Notify an SNS topic (which can then send emails, SMS, etc.)
-   **High Memory Utilization:**
    -   Metric: `MemoryUtilization` (from `AWS/ECS`)
    -   Threshold: e.g., > 80%
    -   Period: e.g., for 15 consecutive minutes
    -   Action: Notify an SNS topic.
-   **Service Task Count Low:**
    -   Metric: `RunningTaskCount` (for your specific service)
    -   Threshold: e.g., < 1 (if `desiredCount` is 1)
    -   Period: e.g., for 5 consecutive minutes
    -   Action: Notify an SNS topic. This is critical for ensuring your service is actually running.

**b. Cost Implication:**
CloudWatch alarms have a small associated cost per alarm per month. The cost is generally low, but it's good to be aware of, especially if creating a large number of alarms.

## 4. CloudWatch Dashboards

Dashboards provide a consolidated view of your most important metrics and logs.

**a. Basic Dashboard:**
Create a CloudWatch Dashboard to visualize:
-   Fargate CPU Utilization (time series graph).
-   Fargate Memory Utilization (time series graph).
-   Running Task Count (number widget or time series).
-   **Log Insights Widget:** Add a widget that queries CloudWatch Logs Insights for specific error messages or critical log patterns from your application (e.g., logs containing "ERROR" or "CRITICAL" from `/ecs/crypto-trading-bot-task`).

Dashboards themselves do not incur extra costs beyond the metrics they display and any Log Insights queries they run. They are a powerful tool for at-a-glance monitoring.

By implementing these CloudWatch features thoughtfully, you can achieve robust logging and monitoring for your crypto trading bot while keeping AWS costs under control. Regularly review your log volumes and custom metric usage to ensure continued cost-effectiveness.
```
