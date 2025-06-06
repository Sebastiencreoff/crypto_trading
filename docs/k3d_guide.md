# Local Development Guide with k3d

This guide outlines how to set up a local Kubernetes development environment using k3d (Rancher's lightweight Kubernetes distribution) for the crypto trading application.

## 1. Prerequisites

*   **Docker Desktop:** Ensure Docker Desktop is installed and running. k3d uses Docker to run the Kubernetes cluster nodes.
*   **k3d:** Install k3d. Follow the instructions at [https://k3d.io/v5/#installation](https://k3d.io/v5/#installation).
*   **kubectl:** Install kubectl, the Kubernetes command-line tool. Instructions at [https://kubernetes.io/docs/tasks/tools/install-kubectl/](https://kubernetes.io/docs/tasks/tools/install-kubectl/).
*   **Local Codebase:** Clone the crypto trading application repository to your local machine.
*   **Environment Variables:** Some scripts or configurations might rely on environment variables (e.g., `SLACK_BOT_TOKEN`). Ensure these are set in your shell or `.env` file if needed for image building or local testing outside Kubernetes.

## 2. k3d Cluster Setup

Create a new k3d cluster. You can customize the cluster name and other parameters. Exposing ports is useful for accessing services later.

```bash
# Create a cluster named 'crypto-dev' and expose port 8080 on localhost
# to forward to the trading service later (mapped to port 80 on the ingress).
k3d cluster create crypto-dev --api-port 6550 -p "8080:80@loadbalancer" --agents 1

# Set kubectl context to the new cluster (usually done automatically by k3d)
kubectl config use-context k3d-crypto-dev
```

Verify the cluster is running:
```bash
kubectl cluster-info
kubectl get nodes
```

## 3. Build Docker Images

The application consists of two main Docker images:
*   **Trading Service:** Handles API requests, manages tasks. (Uses `infra/Dockerfile.trading_service`)
*   **Trading Task:** Runs individual trading algorithms. (Uses `infra/Dockerfile.trading_task`)

Navigate to the root of the repository.

```bash
# Define your Docker Hub username or private registry prefix (replace 'yourusername')
DOCKER_USERNAME=yourusername

# Build the trading service image
docker build -t ${DOCKER_USERNAME}/trading-service:latest -f infra/Dockerfile.trading_service .

# Build the trading task image
docker build -t ${DOCKER_USERNAME}/trading-task:latest -f infra/Dockerfile.trading_task .
```

**Note:** If you have a private registry, ensure you are logged in (`docker login your.registry.com`). Update image tags as needed (e.g., with version numbers).

## 4. Load Images into k3d

Load the newly built Docker images into your k3d cluster. This makes them available to the Kubernetes nodes without needing to push to a remote registry.

```bash
k3d image import ${DOCKER_USERNAME}/trading-service:latest -c crypto-dev
k3d image import ${DOCKER_USERNAME}/trading-task:latest -c crypto-dev
```

## 5. Namespace Setup (Optional)

Using a dedicated namespace is good practice.

```bash
kubectl create namespace crypto-trading
kubectl config set-context --current --namespace=crypto-trading
```
All subsequent `kubectl` commands will apply to this namespace unless specified otherwise. If you don't set the namespace in the context, add `-n crypto-trading` to your `kubectl` commands.

## 6. Kubernetes Secrets

Create Kubernetes secrets for sensitive data. **Do NOT commit real secret values to Git.**

### a. PostgreSQL Credentials

```bash
kubectl create secret generic postgres-credentials \
  --from-literal=POSTGRES_USER=youruser \
  --from-literal=POSTGRES_PASSWORD=yoursecurepassword \
  --from-literal=POSTGRES_DB=cryptodb
```
Replace `youruser`, `yoursecurepassword`, and `cryptodb` with your desired values. These must match what PostgreSQL expects and what the application's `central_config.json` will use.

### b. Binance API Keys (if applicable)

If you plan to run live trading tasks with Binance, create a secret for API keys.

```bash
kubectl create secret generic binance-api-keys \
  --from-literal=BINANCE_API_KEY=your_binance_api_key \
  --from-literal=BINANCE_API_SECRET=your_binance_api_secret
```
**Note:** For local simulation, this might not be strictly necessary if the application handles missing keys gracefully for simulated exchanges.

### c. Slack Bot Token

The `SlackNotifier` uses the `SLACK_BOT_TOKEN` environment variable. This should be provided to the `trading-service` deployment.

```bash
kubectl create secret generic slack-bot-token \
  --from-literal=SLACK_BOT_TOKEN='your_slack_xoxb_token'
```

## 7. Kubernetes ConfigMap

The application uses a ConfigMap to store `central_config.json`.
The provided `infra/kubernetes/configmap.yaml` can be used as a template.

**Key modification needed for k3d:**
*   Update the PostgreSQL host in `central_config.json` within the ConfigMap. In k3d, services are typically accessible within the cluster using `servicename.namespace.svc.cluster.local` or just `servicename` if in the same namespace. For PostgreSQL deployed via StatefulSet (next step), this will be `postgres-service.crypto-trading.svc.cluster.local` (if using `crypto-trading` namespace) or simply `postgres-service`.

Create a copy of `infra/kubernetes/configmap.yaml`, let's say `infra/kubernetes/configmap-k3d.yaml`.
Modify `infra/kubernetes/configmap-k3d.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  # namespace: crypto-trading # ensure this is set if you created a namespace
data:
  central_config.json: |
    {
        "service_name": "crypto_trader_main_k3d",
        "database": {
            "type": "postgresql",
            # IMPORTANT: Update this host
            "host": "postgres-service.crypto-trading.svc.cluster.local", # Or "postgres-service" if in same namespace
            "port": 5432,
            "name": "cryptodb" # Should match POSTGRES_DB from secret
        },
        "exchanges": [
            {
                "name": "binance",
                "extra_settings": {
                    "simulation": true, // For local dev, simulation is safer
                    // API key/secret can be omitted if simulation is true
                    // and the application handles it. Otherwise, reference secrets
                    // or ensure the task runner can access them.
                    "api_key_secret_name": "binance-api-keys", // Optional: if tasks need to fetch these
                    "api_key_key": "BINANCE_API_KEY",
                    "api_secret_key": "BINANCE_API_SECRET"
                }
            }
        ],
        "slack": {
            # This references the key in the 'slack-bot-token' secret
            # The application code (SlackNotifier) expects SLACK_BOT_TOKEN as env var
            # So, the 'slack_bot_token_secret_name' is illustrative if app were to fetch it directly
            # For now, ensure the trading-service deployment sources SLACK_BOT_TOKEN from the secret.
            "default_channel_id": "YOUR_SLACK_CHANNEL_ID_PLACEHOLDER"
        },
        "algorithms": [ /* ... your algo configs ... */ ],
        "other_settings": { /* ... other settings ... */ }
    }
  # Environment variables that will be available to containers using this ConfigMap
  APP_CONFIG_FILE_PATH: "/app/config/central_config.json"
  PYTHONUNBUFFERED: "1"
  # SLACK_BOT_TOKEN will be sourced from a Secret in the Deployment
```
Replace `YOUR_SLACK_CHANNEL_ID_PLACEHOLDER` and other placeholders.

Apply the ConfigMap:
```bash
kubectl apply -f infra/kubernetes/configmap-k3d.yaml
```

## 8. Deploy PostgreSQL

Deploy a PostgreSQL instance within the cluster. We'll use a simple StatefulSet and Service.
You can use `infra/kubernetes/postgres_deployment.yaml` as a starting point. Ensure the environment variables for user, password, and DB name in the StatefulSet match the `postgres-credentials` secret.

Example `postgres-k3d.yaml` (simplified from a typical production setup):
```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  # namespace: crypto-trading
spec:
  ports:
  - port: 5432
  selector:
    app: postgres
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  # namespace: crypto-trading
spec:
  serviceName: "postgres-service"
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13-alpine # Or your preferred version
        ports:
        - containerPort: 5432
        envFrom:
        - secretRef:
            name: postgres-credentials # Matches the secret created earlier
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 1Gi # Adjust storage as needed
```

Apply the PostgreSQL deployment:
```bash
kubectl apply -f postgres-k3d.yaml # Or your equivalent file
```
Check if PostgreSQL pod is running:
```bash
kubectl get pods -l app=postgres
```
Wait for it to be in the `Running` state.

## 9. Deploy Trading Service

Deploy the Trading Service. Use `infra/kubernetes/trading_service_deployment.yaml` as a template.
**Key modifications:**
*   Update `spec.template.spec.containers[0].image` to `${DOCKER_USERNAME}/trading-service:latest`.
*   Ensure the `env` section correctly sources `SLACK_BOT_TOKEN` from the `slack-bot-token` secret.
*   Ensure `envFrom` includes `configMapRef: name: app-config` to load `APP_CONFIG_FILE_PATH`.
*   Database credentials for the service (if it initializes the DB schema) should be passed as environment variables, sourced from the `postgres-credentials` secret.

Example `trading-service-k3d.yaml` (adapt from the existing file):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-service
  # namespace: crypto-trading
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-service
  template:
    metadata:
      labels:
        app: trading-service
    spec:
      containers:
      - name: trading-service
        image: yourusername/trading-service:latest # IMPORTANT: Update this
        ports:
        - containerPort: 8000 # The port FastAPI runs on inside the container
        envFrom:
        - configMapRef:
            name: app-config # For APP_CONFIG_FILE_PATH
        - secretRef:
            name: postgres-credentials # For DB connection from service (e.g., for Alembic migrations)
        env:
        - name: SLACK_BOT_TOKEN
          valueFrom:
            secretKeyRef:
              name: slack-bot-token
              key: SLACK_BOT_TOKEN
        # Add other necessary env vars like DB_USER, DB_PASSWORD, DB_NAME from postgres-credentials
        # if your application directly uses these for DB connection instead of relying solely on a URL from config
        - name: APP_DB_HOST # Example if your app needs these directly
          value: "postgres-service" # Matches the service name of PostgreSQL
        - name: APP_DB_PORT
          value: "5432"
        # Values from postgres-credentials secret:
        - name: APP_DB_USER
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: POSTGRES_USER
        - name: APP_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: POSTGRES_PASSWORD
        - name: APP_DB_NAME # This is POSTGRES_DB in the secret
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: POSTGRES_DB
        readinessProbe:
          httpGet:
            path: /health # Assuming your service has a health check endpoint
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: trading-service-svc # Internal service name
  # namespace: crypto-trading
spec:
  selector:
    app: trading-service
  ports:
  - protocol: TCP
    port: 80 # Port for internal cluster access
    targetPort: 8000 # Port on the pod (FastAPI)
  type: ClusterIP # Or LoadBalancer if you want to expose directly (k3d mapping is preferred for local)
```
**Note:** The `k3d cluster create` command with `-p "8080:80@loadbalancer"` maps `localhost:8080` to the k3d load balancer, which then routes to services. If your service is `ClusterIP`, the k3d ingress controller (traefik by default) will handle routing to it if an Ingress resource is defined. For simplicity here, we'll rely on `kubectl port-forward` or the k3d port mapping if the service type is LoadBalancer or an Ingress is set up.

Apply the Trading Service deployment:
```bash
kubectl apply -f trading-service-k3d.yaml # Or your adapted file
```

## 10. RBAC for Task Manager

The `TaskManager` (running within the `trading-service` pod) needs permissions to create, monitor, and delete Kubernetes Jobs (for trading tasks). This requires a `ServiceAccount`, `ClusterRole`, and `ClusterRoleBinding`.
Use the `infra/kubernetes/task_manager_rbac.yaml` file.

```bash
kubectl apply -f infra/kubernetes/task_manager_rbac.yaml
```
Ensure the `ServiceAccount` name in `task_manager_rbac.yaml` (`task-manager-sa`) is assigned to the `trading-service` Deployment's pods:
```yaml
# In your trading-service-k3d.yaml (or equivalent)
# spec:
#   template:
#     spec:
#       serviceAccountName: task-manager-sa # Add this line
#       containers:
#       ...
```
If you modify the deployment YAML, re-apply it: `kubectl apply -f trading-service-k3d.yaml`. Pods will be recreated.

## 11. Verification

Check if pods are running:
```bash
kubectl get pods
# Look for postgres and trading-service pods in Running state
```

Check logs:
```bash
# Get trading-service pod name
TRADING_SERVICE_POD=$(kubectl get pods -l app=trading-service -o jsonpath='{.items[0].metadata.name}')

kubectl logs $TRADING_SERVICE_POD
kubectl logs -l app=postgres # For PostgreSQL logs
```
Look for successful initialization messages and any error messages.

## 12. Accessing the Service

### Using `kubectl port-forward` (if service is ClusterIP)

```bash
kubectl port-forward svc/trading-service-svc 8000:80
# This forwards localhost:8000 to port 80 of the trading-service-svc
```
Now you can access the API at `http://localhost:8000`.

### Using k3d port mapping (if service is LoadBalancer or Ingress is used)

If you used `k3d cluster create -p "8080:80@loadbalancer"` and your service is exposed correctly (e.g. via LoadBalancer type or an Ingress routing to port 80 of `trading-service-svc`), you should be able to access the API at `http://localhost:8080`.

### Example API Call (create task)

```bash
curl -X POST http://localhost:8000/tasks \
-H "Content-Type: application/json" \
-d '{
    "currency_pair": "BTC/USD",
    "exchange_name": "binance",
    "algo_name": "default_algo",
    "transaction_amount": 100.0
}'
```
(Adjust port if using k3d's 8080 mapping).
You should get a JSON response with a `task_id`.

## 13. Interacting with Tasks (Kubernetes Jobs)

When you create a task via the API, the `TaskManager` creates a Kubernetes Job.

List jobs:
```bash
kubectl get jobs
```
You should see jobs corresponding to the tasks you created.

Get logs from a task pod:
```bash
# Find the pod associated with a job (job names usually contain the task_id)
JOB_POD_NAME=$(kubectl get pods -l job-name=<your-job-name> -o jsonpath='{.items[0].metadata.name}')
kubectl logs $JOB_POD_NAME
```
Replace `<your-job-name>` with the actual job name. The job name will be derived from the `task_id`. The image for these job pods should be `${DOCKER_USERNAME}/trading-task:latest`.

## 14. Cleanup

To delete resources:
```bash
# Delete deployments, services, etc. (use your YAML file names)
kubectl delete -f trading-service-k3d.yaml
kubectl delete -f postgres-k3d.yaml
kubectl delete -f infra/kubernetes/configmap-k3d.yaml
kubectl delete -f infra/kubernetes/task_manager_rbac.yaml
kubectl delete secret postgres-credentials binance-api-keys slack-bot-token

# Delete the namespace if you created one
# kubectl delete namespace crypto-trading

# Delete the k3d cluster
k3d cluster delete crypto-dev
```

This guide provides a comprehensive setup for local development. Remember to adapt image names, secret values, and configurations to your specific setup.
