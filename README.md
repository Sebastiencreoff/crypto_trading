# CI/CD Framework for Application Deployments

This document outlines a GitOps-centric CI/CD framework designed to standardize and automate application deployments across various platforms like Kubernetes and Azure PaaS. It leverages GitHub Actions for CI/CD workflows and a dedicated GitOps repository (`our-company-deployments` or similar) to manage deployment configurations.

## Core Concepts

This framework is built upon several core concepts:

*   **Single Build, Promote Artifact/Image:** Application artifacts (e.g., `.tar.gz`, `.zip`) and Docker images are built once during the "Build, Test, Publish" (BTP) phase. These immutable artifacts/images are then promoted across different environments (DCI, UAT, PPD, PRD). This ensures consistency and reduces environment-specific build variations.
*   **Technology Agnosticism (Generic Core + Custom Parts):** The framework provides a generic core structure and CI/CD workflows. Application-specific build, packaging, and deployment logic are handled by custom scripts (`.cicd/*.sh`) and `Dockerfile`s within each application's source code repository. This allows flexibility for different technology stacks.
*   **Environment Definitions (Environment = Single Application Instance):** An "environment" in this context refers to a specific deployed instance of a single application in a particular stage (e.g., `my-app-dci`, `my-app-uat`). All configuration for such an instance is managed within this GitOps repository.

## Repository Structure (`our-company-deployments`)

This GitOps repository is the central source of truth for all application deployment configurations. Its structure is as follows:

*   **`apps/`**: Contains base deployment templates and type definitions for all managed applications.
    *   `apps/<application_name>/type.txt`: A file indicating the application type (e.g., `kubernetes`, `webapp`, `functionapp`). This dictates which deployment mechanisms are used.
    *   `apps/<application_name>/kubernetes/base/`: Base Kustomize configurations or Helm chart for Kubernetes applications.
    *   `apps/<application_name>/webapp/templates/`: Base ARM/Bicep templates for Azure WebApp applications.
    *   `apps/<application_name>/functionapp/templates/`: Base ARM/Bicep templates for Azure FunctionApp applications.
*   **`environment_instances/`**: Holds configurations for specific deployed instances of applications in various stages.
    *   `environment_instances/<application_name>-<stage>/version.yaml`: Specifies the deployed artifact version or Docker image tag for this instance. **Managed by CI/CD pipelines.**
    *   `environment_instances/<application_name>-<stage>/config_pointers.yaml`: Contains pointers to external configuration services like Azure App Configuration endpoint and Key Vault URI for this instance.
    *   `environment_instances/<application_name>-<stage>/deployment_config/`: Contains instance-specific overrides and configurations:
        *   For Kubernetes: Kustomize overlays/patches (e.g., `kustomization.yaml`, `deployment-patch.yaml`) or environment-specific Helm values files.
        *   For Azure PaaS: ARM/Bicep parameter files (e.g., `azuredeploy.parameters.json`).
*   **`.github/workflows/`**: Contains the GitHub Actions workflow definitions for CI/CD.
    *   `build-test-publish.yml`: Workflow for building, testing, publishing artifacts/images, and performing initial deployment.
    *   `promote-environment.yml`: Workflow for promoting an existing build to a new environment/stage.
*   **`docs/examples/`**: Contains example scripts and configuration files to help development teams onboard their applications.

*(Previously: "GitOps Repository Structure", "Directory Structure")*

## Naming Conventions

Consistent naming is crucial for automation and clarity.

### Artifact Naming
Application artifacts (e.g., from `app-build.sh`) should follow this pattern:
`appname-version-githash.tar.gz` (or other suitable archive format like `.zip`)
*   `appname`: The name of the application.
*   `version`: The application version (e.g., from `pom.xml`, `package.json`, Git tag).
*   `githash`: The short Git commit SHA from which the artifact was built.

*(Previously: "Artifact & Docker Image Naming Conventions" -> "Application Artifact")*

### Docker Image Naming
Docker images should be tagged as follows:
*   **Primary Tag:** `registry/organization/appname:version-githash`
    *   Example: `ghcr.io/myorg/myapi:1.2.0-a1b2c3d`
    *   This is the immutable tag used for promotions.
*   **Environment-Specific Tag (optional):** `registry/organization/appname:<appname>-<stage>-latest`
    *   Example: `ghcr.io/myorg/myapi:myapi-dci-latest`
    *   This tag can be optionally updated to point to the `version-githash` tag currently deployed in a specific stage.

*(Previously: "Artifact & Docker Image Naming Conventions" -> "Docker Image")*

## CI/CD Workflows

The automation heart of this framework lies in its GitHub Actions workflows, located in `.github/workflows/`.

### Workflow 1: Build, Test, Publish & Initial Deploy (`build-test-publish.yml`)
This workflow is responsible for taking application source code, building it, testing it, publishing artifacts/images, and (optionally) performing an initial deployment to a development or integration stage.

*   **Triggers:**
    *   Push to `develop` branch.
    *   Push to `release/*` branches.
    *   Manual trigger (`workflow_dispatch`) with inputs.
*   **Inputs (for manual trigger):**
    *   `application_name`: (string, required) Name of the application.
    *   `initial_deploy_stage`: (string, required, default: `dci`) The stage for initial deployment.
    *   `app_version`: (string, optional) Manual version override.
*   **Key Steps & Customization Hooks:**
    1.  Checkout application code.
    2.  Setup build environment (e.g., Java, Node).
    3.  **Determine Version:** Uses `inputs.app_version` or calls `.cicd/get-version.sh` (team-provided script in app repo) to determine `APP_VERSION`. Generates `GIT_HASH`.
    4.  **Execute Custom App Build:** Runs `sh .cicd/app-build.sh` (team-provided). This script MUST output artifacts to a `./dist` directory in the app repo.
    5.  Archive `./dist` content as `ARTIFACT_NAME` (`appname-APP_VERSION-GIT_HASH.tar.gz`).
    6.  Publish artifact to Artifactory/GitHub Packages (placeholder for specific action).
    7.  **(If Docker app type)** **Execute Custom Docker Build:** Runs `sh .cicd/docker-build.sh` (team-provided) with artifact info, app name, version, hash, and registry prefix. This script builds and tags the Docker image (e.g., `registry/org/appname:APP_VERSION-GIT_HASH`).
    8.  Push Docker image to registry (placeholder for specific action).
    9.  **Update GitOps Repository:** Checks out this GitOps repository, updates `environment_instances/<APP_NAME>-<INITIAL_DEPLOY_STAGE>/version.yaml` with the new image tag or artifact URL. Commits and pushes the change.

### Workflow 2: Promote Environment (`promote-environment.yml`)
This workflow promotes an existing, validated build from a source environment/stage to a target environment/stage by updating the GitOps repository. For non-Kubernetes applications (Azure WebApp, FunctionApp), it can also trigger the deployment.

*   **Triggers:**
    *   Manual trigger (`workflow_dispatch`) with inputs.
*   **Inputs:**
    *   `application_name`: (string, required) The application to promote.
    *   `source_stage`: (string, required) The stage to promote from (e.g., `dci`, `uat`).
    *   `target_stage`: (string, required) The stage to promote to (e.g., `uat`, `ppd`, `prd`).
    *   `version_to_promote`: (string, optional) Specific version/tag to promote. If empty, uses the version currently in `source_stage`.
*   **Promotion Logic & Deployment Dispatching:**
    1.  Checkout this GitOps repository.
    2.  Determine Version: If `version_to_promote` is given, use it. Otherwise, read `image_tag` and `artifact_url` from `environment_instances/<APP_NAME>-<SOURCE_STAGE>/version.yaml`.
    3.  Read Application Type: Reads `apps/<APP_NAME>/type.txt`.
    4.  **(Conditional) PRD Approval:** If `TARGET_STAGE` is `prd`, the workflow requires manual approval via a GitHub Actions Environment named `prd`.
    5.  **Update GitOps Repository:** Updates `environment_instances/<APP_NAME>-<TARGET_STAGE>/version.yaml` with the version to promote. Commits and pushes.
    6.  **Deployment Triggering (for non-Kubernetes):**
        *   **Kubernetes:** ArgoCD (or similar GitOps agent) automatically syncs the changes from the GitOps repo. The workflow just logs this.
        *   **Azure WebApp/FunctionApp:** The workflow logs into Azure (using `AZURE_CREDENTIALS` secret), fetches relevant deployment configurations (base ARM templates from `apps/`, instance parameters from `environment_instances/.../deployment_config/`), and uses Azure CLI or specific GitHub Actions (e.g., `azure/webapps-deploy@v2`) to deploy/update the PaaS service. It injects artifact/image details from `version.yaml` and configuration pointers from `config_pointers.yaml` as needed.
    7.  **(Optional) Post-Promotion Hook:** Executes `.cicd/post-promote.sh` (team-provided in app repo) for custom actions.

## Developer Team Responsibilities

Successful adoption of this framework requires collaboration between platform/DevOps teams and application development teams. Development teams have the following key responsibilities:

### Application Source Code Repository:
This is the repository containing the application's actual source code.
*   **Custom Build Script (`.cicd/app-build.sh`)**
    *   **Responsibilities:** Each application repository **must** provide this script. It's responsible for the entire build lifecycle: compiling, testing, and packaging the application into deployable artifact(s).
    *   **Artifact Output:** Critically, this script **must** place all final deployable artifacts into a directory named `./dist` at the root of the application repository. The BTP workflow archives this directory.
    *   **Example:** [`docs/examples/cicd_scripts/app-build.sh.example`](./docs/examples/cicd_scripts/app-build.sh.example)
*(Previously: "Custom Build Script (`.cicd/app-build.sh`)")*
*   **Custom Docker Build (`.cicd/docker-build.sh` and `Dockerfile`)**
    *   For applications to be deployed as Docker containers, the application repository **must** provide:
        1.  A `Dockerfile` (typically at the root).
        2.  A `.cicd/docker-build.sh` script.
    *   **`Dockerfile` Responsibilities:** Defines base image, artifact copying, setup, runtime command.
    *   **`.cicd/docker-build.sh` Responsibilities:** Accepts parameters (artifact info, app name, version, hash, registry prefix) from the BTP workflow, prepares the artifact for Docker build, executes `docker build` (passing build args), and tags the image as per conventions.
    *   **Examples:**
        *   [`docs/examples/cicd_scripts/Dockerfile.example`](./docs/examples/cicd_scripts/Dockerfile.example)
        *   [`docs/examples/cicd_scripts/docker-build.sh.example`](./docs/examples/cicd_scripts/docker-build.sh.example)
*(Previously: "Custom Docker Build Script (`.cicd/docker-build.sh`) and `Dockerfile`")*

### GitOps Repository Contributions (`our-company-deployments`):
This is the central repository described in this document.
*   **Adding New Applications:**
    *   Define the base configuration in `apps/<your-app-name>/`. This includes `type.txt` and relevant subdirectories (`kubernetes/base/`, `webapp/templates/`, etc.) with base Kustomize files or ARM templates.
*   **Configuring Environment Instances:**
    *   Create the initial directory structure for new instances (e.g., `environment_instances/<your-app-name>-dci/`).
    *   Populate the initial `config_pointers.yaml` with the correct URIs/endpoints for Azure App Configuration and Key Vault for that specific instance.
    *   Provide instance-specific `deployment_config/` files (Kustomize overlays, ARM parameter files) that customize the base templates from `apps/`.
*(Partially from previous "Deployment Manifests and Configuration")*

### Azure Resource Management:
*   **Provisioning:** Create and manage the actual Azure App Configuration stores and Azure Key Vaults needed for each application instance (e.g., `my-app-dci-appconfig`, `my-app-dci-kv`).
*   **Populating:** Add the necessary key-values, feature flags, and secrets into these Azure services for each environment. Application code will read from these.

*(Previously: "Application Configuration with Azure App Configuration and Key Vault" -> "Developer Team Responsibilities")*

## Application Configuration Strategy

### Using Azure App Configuration and Key Vault
This framework promotes managing application configuration and secrets externally using Azure App Configuration and Azure Key Vault. This approach enhances security, allows for dynamic configuration updates, and provides robust environment-specific management. Configuration data itself is **not** stored directly in this GitOps repository.

*   **Role of `config_pointers.yaml`:**
    For each deployed application instance, the `environment_instances/<application_name>-<stage>/config_pointers.yaml` file plays a crucial role. It stores metadata that points to the specific Azure services used by that instance. Example:
    ```yaml
    # environment_instances/my-app-dci/config_pointers.yaml
    azure_app_config_store_endpoint: "https://my-app-dci-appconfig.azconfig.io"
    azure_key_vault_uri: "https://my-app-dci-kv.vault.azure.net/"
    ```
    This file is typically populated manually or by specialized infrastructure provisioning scripts. CI/CD mechanisms use this to configure application access.
*   **Access Methods (Managed Identities, Service Principals):**
    *   **Managed Identities:** **Strongly recommended** for Azure-hosted applications (AKS, WebApps, FunctionApps) for credential-less access.
    *   **Service Principals:** Alternative if Managed Identities are unsuitable.
*   **Integration for Kubernetes (CSI Driver, App Config Provider):**
    *   **Azure Key Vault:** Use the Azure Key Vault Provider for Secrets Store CSI Driver. Define `SecretProviderClass` CRDs referencing the Key Vault URI from `config_pointers.yaml`.
    *   **Azure App Configuration:** Use the Azure App Configuration Kubernetes Provider (watches App Config store via endpoint from `config_pointers.yaml` and creates/updates ConfigMaps) or use Azure SDKs directly in the application (pod identity needed).
*   **Integration for Azure WebApps & FunctionApps (Key Vault References, SDKs):**
    *   **Azure Key Vault:** Utilize Key Vault References in App Settings. The platform resolves these using the PaaS service's Managed Identity (which needs GET permissions on the KV).
    *   **Azure App Configuration:** Applications use Azure SDKs with the store endpoint (from `config_pointers.yaml`, provided as an App Setting). The Managed Identity needs READ permissions.
    *   The `promote-environment.yml` workflow helps set up these App Settings and ensures Managed Identity permissions.

*(Previously: "Application Configuration with Azure App Configuration and Key Vault" - condensed and integrated)*
*(Also includes "Deployment Manifests and Configuration" -> "Integration with Azure App Configuration and Key Vault")*

## Setup & Prerequisites

Before using this framework effectively, ensure the following are in place:

*   **Required GitHub Actions Secrets:**
    These must be configured in the **application source code repository settings** (Settings > Secrets and variables > Actions) for the CI/CD pipelines to function:
    *   `ARTIFACTORY_URL`: (If using Artifactory for artifacts) Base URL of Artifactory.
    *   `ARTIFACTORY_USER`: Username for Artifactory.
    *   `ARTIFACTORY_TOKEN`: Access token/password for Artifactory.
    *   `GITHUB_TOKEN`: (If using GitHub Packages) Usually available, note its usage.
    *   `DOCKER_REGISTRY_URL`: URL of your Docker registry (e.g., `ghcr.io`, `youracr.azurecr.io`).
    *   `DOCKER_REGISTRY_USER`: Username for the Docker registry.
    *   `DOCKER_REGISTRY_PASSWORD`: Password/token for the Docker registry.
    *   `AZURE_CREDENTIALS`: JSON output of `az ad sp create-for-rbac --sdk-auth` for Azure deployments/management.
    *   `GITOPS_REPO_PAT`: A Personal Access Token with write access to *this* GitOps repository, used by workflows to push `version.yaml` updates. *(New addition based on workflow needs)*
*(Previously: "Prerequisites and Setup" -> "Required GitHub Actions Secrets")*

*   **Artifact Repository Setup:**
    *   An artifact repository (JFrog Artifactory, GitHub Packages, Azure Artifacts, etc.) is needed to store build artifacts.
    *   Ensure repositories/feeds are created and CI has credentials.
*(Previously: "Prerequisites and Setup" -> "Artifact Repository Setup")*

*   **Docker Registry Setup:**
    *   A Docker registry (JFrog Artifactory, GHCR, ACR, Docker Hub, etc.) is required for Docker images.
    *   Ensure repositories exist and CI has credentials.
*(Previously: "Prerequisites and Setup" -> "Docker Registry Setup")*

## Getting Started: Onboarding a New Application

Here's a checklist for onboarding a new application to this framework:

1.  **Understand Framework:** Review this README document thoroughly.
2.  **Application Source Code Repository Setup:**
    *   Implement the `.cicd/app-build.sh` script to compile, test, and package your application, placing artifacts in a `./dist` directory.
    *   If your application will be deployed as a Docker container:
        *   Create a `Dockerfile`.
        *   Implement the `.cicd/docker-build.sh` script to build and tag your Docker image, using the artifact from `app-build.sh`.
3.  **GitOps Repository Setup (This Repository - `our-company-deployments`):**
    *   **Add Base Configuration:**
        *   Create a new directory `apps/<your-app-name>/`.
        *   Inside, create `type.txt` with the correct type (`kubernetes`, `webapp`, or `functionapp`).
        *   Add base Kustomize configurations (in `kubernetes/base/`), ARM templates (in `webapp/templates/` or `functionapp/templates/`), or other base manifest files.
    *   **Configure Initial Environment Instance:**
        *   Create `environment_instances/<your-app-name>-<initial-stage>/` (e.g., `-dci`).
        *   In this directory, create `config_pointers.yaml` and populate it with the actual Azure App Configuration endpoint and Key Vault URI for this specific instance.
        *   Create `environment_instances/<your-app-name>-<initial-stage>/deployment_config/`.
        *   Add instance-specific configurations here (e.g., `kustomization.yaml` and patches for Kubernetes, or `azuredeploy.parameters.json` for Azure PaaS). These should customize the base configurations from `apps/`.
        *   Ensure `version.yaml` is either absent (will be created by CI) or contains initial placeholder values.
4.  **GitHub Secrets Configuration:**
    *   In your **application's source code repository** (not this GitOps repo), go to Settings > Secrets and variables > Actions.
    *   Configure all secrets listed under "Required GitHub Actions Secrets" (e.g., `ARTIFACTORY_URL`, `DOCKER_REGISTRY_USER`, `AZURE_CREDENTIALS`, `GITOPS_REPO_PAT`).
5.  **Initial Deployment:**
    *   Trigger the "Build, Test, Publish & Initial Deploy" (`build-test-publish.yml`) workflow from your application's repository. Provide the required inputs (application name, initial deploy stage).
    *   This will build your application, publish the artifact/image, and update the `version.yaml` in this GitOps repository for your initial stage, triggering the deployment via ArgoCD (for Kubernetes) or direct action (for PaaS, if configured in the BTP workflow - though typically initial PaaS deploy might be manual or via promote).
6.  **Subsequent Promotions:**
    *   Once the initial deployment is verified, use the "Promote Environment" (`promote-environment.yml`) workflow (triggered from this GitOps repository or the app repo, depending on setup) to promote the application to higher environments (e.g., UAT, PRD).

## Examples in this Repository
This repository contains examples to guide you:

*   **Application Base Configurations:**
    *   [`apps/my-kubernetes-app/`](./apps/my-kubernetes-app/): Example for a Kubernetes application.
    *   [`apps/my-webapp/`](./apps/my-webapp/): Example for an Azure WebApp.
    *   [`apps/my-function-app/`](./apps/my-function-app/): Example for an Azure Function App.
*   **Environment Instance Configurations:**
    *   [`environment_instances/my-kubernetes-app-dci/`](./environment_instances/my-kubernetes-app-dci/)
    *   [`environment_instances/my-kubernetes-app-uat/`](./environment_instances/my-kubernetes-app-uat/)
    *   [`environment_instances/my-webapp-dci/`](./environment_instances/my-webapp-dci/)
    *   [`environment_instances/my-function-app-dci/`](./environment_instances/my-function-app-dci/)
*   **CI/CD Script Examples:**
    *   [`docs/examples/cicd_scripts/app-build.sh.example`](./docs/examples/cicd_scripts/app-build.sh.example)
    *   [`docs/examples/cicd_scripts/docker-build.sh.example`](./docs/examples/cicd_scripts/docker-build.sh.example)
    *   [`docs/examples/cicd_scripts/Dockerfile.example`](./docs/examples/cicd_scripts/Dockerfile.example)

*(Previously: "Deployment Manifests and Configuration" -> "Examples")*
*(Also includes "Artifact Output Location for Custom Scripts" - integrated into Developer Responsibilities)*
*(Original "GitOps Repository Structure" and "Directory Structure" are now "Repository Structure (`our-company-deployments`)")*
