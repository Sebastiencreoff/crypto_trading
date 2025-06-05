# Cost-Optimized Architecture for Crypto Trading Bot on AWS

## 1. Introduction

The goal of this architecture is to deploy the crypto trading bot in a manner that is functional, secure, and cost-optimized on AWS. It synthesizes choices made during the planning phase to minimize operational expenses without compromising core requirements. For initial deployment, manual processes are favored to defer CI/CD costs.

## 2. Core Architectural Components and Cost Optimizations

This architecture leverages several AWS services and strategies to achieve cost efficiency:

*   **Source Code Management:**
    *   **Choice:** GitHub (free for private repositories with sufficient features) or AWS CodeCommit (offers a free tier).
    *   **Cost Optimization:** Utilizing free tiers or existing low-cost solutions for source control.

*   **Containerization (Dockerfile):**
    *   **Strategy:** The `Dockerfile` is optimized to use a slim Python base image (`python:3.9-slim-buster`) and efficient layering.
    *   **Cost Optimization:** Smaller image sizes reduce storage costs in ECR and can lead to slightly faster startup times.

*   **Container Registry (Amazon ECR):**
    *   **Strategy:** Store the Docker container image in ECR.
    *   **Cost Optimization:** ECR has a free tier for storage, and subsequent costs are low. Smaller images (from Dockerfile optimization) further reduce these costs. ECR image scanning is enabled for security at no extra charge for basic scanning.

*   **Application Hosting (AWS Fargate):**
    *   **Strategy:** Run the containerized application using AWS Fargate for serverless container orchestration.
    *   **Cost Optimization:**
        *   **Fargate Spot:** Utilize Fargate Spot for compute capacity, which can offer significant savings (up to 70%) over Fargate On-Demand, suitable for fault-tolerant workloads like this bot (if it can handle interruptions and restart).
        *   **Right-Sizing:** The task definition specifies a small CPU (`256` / 0.25 vCPU) and memory (`512` MB) allocation, aligned with the application's expected needs. This should be monitored and adjusted.
        *   **Single Instance:** The service is configured for a `desiredCount` of 1, minimizing constant resource use.

*   **Database (Conditional):**
    *   **Strategy:**
        *   **SQLite (Default):** The application defaults to using SQLite with the database file stored within the container's ephemeral storage or a small, inexpensive EFS volume if persistence across restarts is strictly needed (though this adds complexity and some cost). For a simple bot, ephemeral storage might be acceptable if its state can be reconstructed or is not critical.
        *   **Amazon RDS (Optional):** If a managed PostgreSQL database is required (e.g., for more complex data analysis or state management), `psycopg2-binary` is included. RDS would be an additional cost component (instance hours, storage, data transfer). For maximum cost savings, SQLite is preferred.
    *   **Cost Optimization:** Avoiding RDS unless absolutely necessary. If SQLite is used, ephemeral storage is free; EFS would have a small cost.

*   **Configuration & Secrets Management:**
    *   **Strategy:** Application configuration files (e.g., `trading_COINBASE.json`, `trading_SIMU.json`) are included in the Docker image. Sensitive data like API keys and database credentials (if using RDS) are stored in AWS Secrets Manager.
    *   **Cost Optimization:** Secrets Manager has a low cost per secret per month and per 10,000 API calls. This is an essential security measure with a justifiable, minimal cost.

*   **Networking (VPC):**
    *   **Strategy:** The Fargate task runs in a Virtual Private Cloud (VPC).
        *   **Private Subnets:** Tasks are deployed in private subnets to prevent direct inbound internet access, enhancing security.
        *   **Security Groups:** Restrict inbound/outbound traffic to only what's necessary.
        *   **NAT Gateway:** Required for tasks in private subnets to access the internet (e.g., Coinbase API, AWS services).
    *   **Cost Optimization:** While VPC and Security Groups themselves are free, the **NAT Gateway incurs hourly charges and data processing fees.** This is a trade-off for enhanced security. If the bot could operate without outbound internet (e.g., only accessing services within the VPC via VPC Endpoints, which is not the case here for Coinbase), NAT Gateway costs could be avoided. VPC Endpoints for services like ECR and Secrets Manager can reduce data transfer costs over NAT Gateway and improve security.

*   **Logging & Monitoring (Amazon CloudWatch):**
    *   **Strategy:** Centralized logging and basic monitoring using CloudWatch.
    *   **Cost Optimization:**
        *   **Log Retention:** A log retention policy (e.g., 30 days) is set for the CloudWatch Log Group to manage storage costs.
        *   **Metrics & Alarms:** Rely primarily on standard Fargate metrics (CPU/Memory utilization), which are part of the free tier or have predictable costs. Custom metrics and excessive alarms are avoided unless critical, due to their associated costs.

*   **Deployment Process (Initial):**
    *   **Strategy:** Manual Docker build and deployment to ECS/Fargate using AWS CLI or Management Console.
    *   **Cost Optimization:** Avoids costs associated with CI/CD automation services (CodePipeline, CodeBuild) for initial deployment and infrequent updates.

*   **CI/CD Pipeline (Optional Future Enhancement):**
    *   **Strategy:** A full CI/CD pipeline (e.g., using AWS CodePipeline, CodeBuild) is documented as an option for when deployments become more frequent.
    *   **Cost Optimization:** Deferring implementation of a full CI/CD pipeline saves on the costs of these services until the operational benefits outweigh the expenses.

## 3. Conceptual Diagram/Flow

```
                                       +---------------------+
                                       |   Source Control    |
                                       | (GitHub/CodeCommit) |
                                       +---------------------+
                                                 | (Manual Trigger for CI/CD, or Manual Build)
                                                 v
+----------------------+       +---------------------+       +-----------------------+
| Developer/Maintainer |------>|  Local Docker Build |------>| Amazon ECR            |
| (Manual Steps)       |       +---------------------+       | (Container Registry)  |
+----------------------+                                     +-----------------------+
                                                                         |
                                                                         v
                                                         +---------------------------+
                                                         | AWS Fargate / Amazon ECS  |
                                                         | (crypto-trading-bot task)|
                                                         |  - Fargate Spot           |
                                                         |  - Right-sized CPU/Mem    |
                                                         |  - Private Subnet         |
                                                         +---------------------------+
                                                               |       ^       |
                                                               |       |       | (Secrets)
                                     (Outbound API calls)      v       |       v
 +-------------------------+       +---------------------+   +---------------------+
 | Internet (Coinbase API) |<------| VPC NAT Gateway     |<--| AWS Secrets Manager |
 +-------------------------+       | (Public Subnet)     |   +---------------------+
                                   +---------------------+
                                                               |       ^
                                                               |       | (Logs, Metrics)
                                                               v       |
                                                         +---------------------+
                                                         | Amazon CloudWatch   |
                                                         | - Logs (retention)  |
                                                         | - Metrics (basic)   |
                                                         | - Alarms (minimal)  |
                                                         +---------------------+
```

## 4. Conclusion

This architecture aims to provide a sensible balance between operational functionality, robust security, and cost optimization for the crypto trading bot. By leveraging Fargate Spot, right-sizing resources, implementing log retention, and initially opting for manual deployment processes, AWS operational costs can be significantly minimized. Future enhancements like a CI/CD pipeline can be layered on as the project matures and operational needs evolve. Regular review of resource utilization and AWS billing will be key to maintaining cost efficiency.
```
