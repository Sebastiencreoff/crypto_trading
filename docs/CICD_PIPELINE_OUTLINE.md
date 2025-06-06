# CI/CD Pipeline Outline for Crypto Trading Bot (Optional Enhancement)

## 1. Introduction

**Continuous Integration/Continuous Deployment (CI/CD)** is a set of practices and tools that automate the process of building, testing, and deploying software. Key benefits include:
-   **Automation:** Reduces manual effort and potential for human error.
-   **Consistency:** Ensures every deployment follows the same process.
-   **Faster Releases:** Enables more frequent and reliable updates.

**For initial cost savings and simplicity, the manual deployment process outlined in `DEPLOY_AWS.md` is the recommended starting point for the crypto trading bot.** This document outlines a *potential* CI/CD pipeline as an optional future enhancement if deployment frequency or complexity increases.

## 2. Core AWS Services for a Potential CI/CD Pipeline

A CI/CD pipeline on AWS could leverage the following services:

*   **AWS CodeCommit (or GitHub/Bitbucket/etc.):**
    *   **Description:** A managed source control service (Git).
    *   **Cost:** Has a free tier for a small number of users and repositories; beyond that, it's per user per month. Using existing GitHub/Bitbucket repositories is also common.
*   **AWS CodeBuild:**
    *   **Description:** A fully managed continuous integration service that compiles source code, runs tests, and produces software packages ready to deploy.
    *   **Cost:** Charged per build minute, with different rates based on the compute instance type. A small free tier is available.
*   **AWS CodePipeline:**
    *   **Description:** A fully managed continuous delivery service that automates your release pipelines for fast and reliable application and infrastructure updates.
    *   **Cost:** Charged per active pipeline per month. A pipeline is considered active if it has at least one code change run through it during the month. Free for the first 30 days.
*   **Amazon Elastic Container Registry (ECR):**
    *   **Description:** A managed Docker container registry to store, manage, and deploy Docker container images.
    *   **Cost:** Primarily for storage and data transfer out. A free tier for storage is available.
*   **Amazon Elastic Container Service (ECS) with AWS Fargate:**
    *   **Description:** Used for running the containerized application. CodePipeline can integrate with ECS to automate deployments.
    *   **Cost:** As described in `DEPLOY_AWS.md` (per vCPU and GB memory per second for Fargate).

## 3. Phases of the CI/CD Pipeline

A typical CI/CD pipeline for this application would look like this:

**a. Source Stage:**
*   **Provider:** AWS CodeCommit, GitHub, Bitbucket, etc.
*   **Trigger:** Automatically starts the pipeline when a change is pushed to a specific branch (e.g., `main` or `master`).

**b. Build Stage:**
*   **Provider:** AWS CodeBuild.
*   **Configuration (`buildspec.yml` or directly in CodeBuild project):**
    1.  **Pre-build:** Login to Amazon ECR.
    2.  **Build:**
        *   Run `docker build -t $REPOSITORY_URI:$IMAGE_TAG .` to build the Docker image.
        *   (Optional but Recommended) Run unit tests and integration tests (e.g., `python -m unittest discover` or using `pytest`). A build failure here would stop the pipeline.
    3.  **Post-build:**
        *   Run `docker push $REPOSITORY_URI:$IMAGE_TAG` to push the newly built image to ECR.
        *   Generate an `imagedefinitions.json` file. This file tells CodePipeline's ECS deploy action which image to use. Example content:
            ```json
            [
              {
                "name": "crypto-trading-bot-container", // Must match container name in Task Definition
                "imageUri": "<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/crypto-trading-bot:latest" // Dynamically set this
              }
            ]
            ```
*   **Artifacts:** The `imagedefinitions.json` file is output as a build artifact.

**c. Deploy Stage:**
*   **Provider:** Amazon ECS (via CodePipeline).
*   **Action:** CodePipeline uses the `imagedefinitions.json` file from the build stage to update the specified ECS service (e.g., `crypto-trading-bot-service`). This triggers a new deployment in Fargate, pulling the latest image from ECR and replacing the old tasks.

## 4. Cost Implications and Alternatives

**a. Potential Costs of an AWS CI/CD Pipeline:**
-   **CodeCommit:** Potentially free, or a few dollars per month depending on users.
-   **CodeBuild:** Depends on build duration and frequency. For a simple Docker build and test, it might be a few dollars per month if builds are frequent.
-   **CodePipeline:** ~$1 per active pipeline per month (after the free tier or if used previously).
-   **ECR:** Storage costs are generally low for a few images unless many large versions are kept.

While each service is individually cost-effective, the total can add up, especially if deployment frequency is low.

**b. Manual Deployment (Primary Cost-Saving Method):**
As detailed in `DEPLOY_AWS.md`, manually building the Docker image, pushing it to ECR, and updating the ECS service (either via AWS Management Console or AWS CLI by re-registering the task definition and updating the service) incurs minimal costs beyond the standard ECR storage and Fargate compute. This is the most cost-effective approach for infrequent deployments.

**c. Alternatives:**
-   **GitHub Actions:** If your code is hosted on GitHub, GitHub Actions offers a generous free tier for public and private repositories that can be used to build Docker images, push to ECR, and even deploy to ECS (using community-supported actions or AWS CLI scripts).
-   **GitLab CI/CD:** Similarly, if using GitLab, its built-in CI/CD capabilities can achieve the same results, often within its free tier.

These alternatives can be more cost-effective if you are already utilizing these platforms.

## 5. Recommendation

*   **Start with Manual Deployment:** For the crypto trading bot, especially in its early stages or if updates are infrequent, manual deployment is the most cost-effective approach. Follow the steps in `DEPLOY_AWS.md`.
*   **Consider CI/CD Later:** If the application evolves, deployments become more frequent (e.g., multiple times a week), or the deployment process becomes more complex (e.g., involving multiple environments, blue/green deployments), then investing time and resources into setting up a CI/CD pipeline becomes more justifiable. At that point, evaluate the AWS services or alternatives like GitHub Actions based on your existing tools and cost preferences.

By prioritizing manual deployment initially, you can keep operational costs low while still having a clear path for future automation if needed.
```
