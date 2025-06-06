# Manual Testing Guide (for k3d Deployment)

This guide outlines conceptual manual tests for the crypto trading application after it has been deployed to a local k3d cluster using the `docs/k3d_guide.md`.

## Prerequisites

*   The application (PostgreSQL, `trading-service`) is deployed and running in your k3d cluster.
*   You have `kubectl` access to the cluster and the correct namespace is selected.
*   You can access the `trading-service` API (e.g., via `kubectl port-forward` or the k3d load balancer).
*   You have access to the configured Slack workspace and channel to observe notifications.
*   You have a PostgreSQL client to inspect the database if needed.

## I. Trading Service API

These tests focus on the functionality of the `trading-service` FastAPI application.

### 1. Health Check

*   **Test:** Access the health check endpoint (e.g., `/health` or a similar endpoint if implemented) of the `trading-service`.
*   **Expected Outcome:** The service returns a successful response (e.g., HTTP 200 OK), indicating it's running.
*   **Verification:**
    *   Use `curl` or a web browser.
    *   `kubectl logs <trading-service-pod>` should not show critical errors related to startup.

### 2. Task Creation

*   **Test:** Send a POST request to the `/tasks` endpoint to create a new trading task. Use valid parameters for `currency_pair`, `exchange_name` (e.g., "binance" for simulation), `algo_name`, and `transaction_amount`.
*   **Expected Outcome:**
    *   The API returns HTTP 201 Created with a JSON response containing the `task_id`.
    *   A new Kubernetes Job should be created in the cluster.
*   **Verification:**
    *   Note the `task_id` from the response.
    *   `kubectl get jobs`: A new job corresponding to the task should appear (its name often contains the `task_id`).
    *   `kubectl logs <trading-service-pod>`: Logs should indicate successful task submission to `TaskManager` and job creation.

### 3. Invalid Task Creation

*   **Test:** Send POST requests to `/tasks` with invalid data:
    *   Non-existent `exchange_name`.
    *   Non-existent `algo_name`.
    *   Missing required fields.
*   **Expected Outcome:** The API returns appropriate HTTP error codes (e.g., 400 Bad Request, 422 Unprocessable Entity) with informative error messages.
*   **Verification:** Check API response body and status code.

### 4. Get Task Status

*   **Test:** Send a GET request to `/tasks/{task_id}` using a `task_id` obtained from a successful task creation.
*   **Expected Outcome:** The API returns HTTP 200 OK with the task's current status (e.g., "Pending", "Running", "Completed", "Failed").
*   **Verification:**
    *   The status should reflect the state of the Kubernetes Job.
    *   `kubectl get job <job-name>`: Check the job's status conditions.
    *   `kubectl get pods -l job-name=<job-name>`: Check the status of the pod running the task.

### 5. Get Status for Non-existent Task

*   **Test:** Send a GET request to `/tasks/{task_id}` using a fictional `task_id`.
*   **Expected Outcome:** The API returns HTTP 404 Not Found.
*   **Verification:** Check API response status code.

### 6. Stop Task

*   **Test:** Send a POST request to `/tasks/{task_id}/stop` for a currently "Running" task.
*   **Expected Outcome:**
    *   The API returns HTTP 200 OK indicating the stop signal was sent.
    *   The corresponding Kubernetes Job and its pod should eventually terminate.
*   **Verification:**
    *   `kubectl logs <trading-service-pod>`: Logs should indicate the stop request was processed.
    *   `kubectl get job <job-name>`: The job might be marked as completed or show signs of termination.
    *   `kubectl logs <task-pod-name>` (before it terminates): Task logs should show it received a stop signal and is shutting down.
    *   Eventually, `kubectl get pods -l job-name=<job-name>` should show no running pods for that job.

### 7. Stop Non-existent or Already Stopped Task

*   **Test:** Send a POST request to `/tasks/{task_id}/stop` for a fictional `task_id` or a task that has already completed/failed.
*   **Expected Outcome:** The API returns an appropriate error (e.g., HTTP 404 Not Found or a specific status indicating it cannot be stopped).
*   **Verification:** Check API response.

## II. Trading Logic (within Task Container)

These tests focus on the behavior of the code running inside the `trading-task` containers (Kubernetes Job pods).

### 1. Configuration Parsing

*   **Test:** After a task starts, inspect its logs.
*   **Expected Outcome:** The task log should indicate successful parsing of its configuration (exchange settings, algorithm parameters, task parameters like currency pair, amount).
*   **Verification:** `kubectl logs <task-pod-name>`: Look for log entries showing the loaded configuration values.

### 2. Database Interaction (Price Ticks, Transactions)

*   **Test:** While a task is running (especially a simulation task that generates data quickly):
    *   Allow it to simulate a few buy/sell operations if possible, or at least process some price data.
*   **Expected Outcome:**
    *   Price ticks should be saved to the `price_ticks` table in the PostgreSQL database.
    *   If trades occur, new records should appear in the `trading_transactions` table, and existing ones should be updated with sell details and profit.
*   **Verification:**
    *   Connect to the PostgreSQL database (e.g., via `kubectl port-forward service/postgres-service 5432:5432` and a `psql` client).
    *   `SELECT * FROM price_ticks WHERE task_id = 'your_task_id' ORDER BY timestamp DESC LIMIT 10;`
    *   `SELECT * FROM trading_transactions WHERE task_id = 'your_task_id' ORDER BY buy_date_time DESC;`

### 3. Exchange Connection (Simulation)

*   **Test:** For a task configured with a simulated exchange (e.g., Binance simulation).
*   **Expected Outcome:** The task should fetch price data from the simulation source. If it's file-based, ensure it reads the file. If it's generating data, ensure this happens.
*   **Verification:** `kubectl logs <task-pod-name>`: Look for logs indicating:
    *   Connection to the simulation exchange.
    *   Fetching or generation of price data (e.g., "Currency Value for BTC/USD: 50000.0").
    *   Simulated buy/sell order execution messages.

### 4. Algorithm Signal Generation

*   **Test:** Observe task logs for a running task.
*   **Expected Outcome:** The logs should show the algorithm processing data and generating signals (e.g., "AlgoSignal: 1" for buy, "-1" for sell, "0" for hold).
*   **Verification:** `kubectl logs <task-pod-name>`: Filter for "AlgoSignal" or similar log messages.

## III. Integrated Notification System (Slack)

These tests verify that Slack notifications are sent correctly from the `trading-service` and potentially from tasks if they are configured to do so directly (though the current merged architecture centralizes it in `trading-service` which tasks call into).

### 1. Task Creation Notification

*   **Test:** Create a new task via the API.
*   **Expected Outcome:** A Slack message should be posted to the configured default channel announcing the task creation or submission, including the `task_id`.
*   **Verification:** Check the Slack channel.

### 2. Task Start/Stop/Completion/Failure Notifications

*   **Test:** Observe a task through its lifecycle (start, run, manually stop it, or let it complete/fail).
*   **Expected Outcome:** Slack messages should be sent for key events:
    *   Task started running.
    *   Task stopped (either by request or normally).
    *   Task completed successfully.
    *   Task failed (if applicable, due to an error).
*   **Verification:** Check the Slack channel for messages corresponding to these events, referencing the correct `task_id`.

### 3. Trade Execution Notifications

*   **Test:** If a task executes a simulated (or real) buy or sell order.
*   **Expected Outcome:** A Slack message should be sent detailing the trade (e.g., "Bought X BTC at Y price", "Sold X BTC at Z price. Profit: A EUR").
*   **Verification:** Check the Slack channel.

### 4. Error/Critical Event Notifications

*   **Test:** Induce an error if possible (e.g., misconfigure something that a task would pick up, or if a task has a known failure mode).
*   **Expected Outcome:** Critical errors within a task or the main service should trigger a Slack notification.
*   **Verification:** Check the Slack channel.

### 5. Test Slack Notification Endpoint (if still enabled)

*   **Test:** Call the `/debug/notify` endpoint on the `trading-service` API with a test message.
*   **Expected Outcome:** The test message should appear in the default Slack channel.
*   **Verification:** Check Slack.

## IV. Configuration Loading

### 1. Trading Service Configuration

*   **Test:** Check `trading-service` logs upon startup.
*   **Expected Outcome:** Logs should indicate successful loading of `central_config.json` from the path specified by `APP_CONFIG_FILE_PATH` (mounted from the ConfigMap). Database connection details and Slack configuration should be correctly parsed.
*   **Verification:** `kubectl logs <trading-service-pod>`: Look for logs related to configuration loading and Slack/TaskManager initialization.

### 2. Trading Task Configuration

*   **Test:** Check logs of a newly started `trading-task` pod.
*   **Expected Outcome:** The task pod logs should show the specific configuration it received (exchange settings, algo settings, task parameters like currency pair) which was passed to it by the `TaskManager` when the job was created.
*   **Verification:** `kubectl logs <task-pod-name>`: Look for initialization logs showing these parameters.

## V. Persistence (PostgreSQL)

This mostly overlaps with "Database Interaction" under Trading Logic but focuses on data integrity over time.

### 1. Data Integrity Across Task Restarts (Conceptual)

*   **Test:** If a task is stopped and a new one is started for the same purpose (e.g., same currency pair, if the system is designed to resume or work with past data).
*   **Expected Outcome:** New data should be appended correctly. If the system has logic to prevent re-processing or to consider previous state, this should function as expected. (This is highly dependent on application logic not explicitly detailed).
*   **Verification:** Inspect database tables (`price_ticks`, `trading_transactions`) to see if data from different task instances is stored correctly and without unintended overwrites or corruption.

### 2. Data Consistency

*   **Test:** After several trades, query the `trading_transactions` table.
*   **Expected Outcome:**
    *   Buy transactions should have buy prices and amounts.
    *   Sell transactions should correctly link to buy transactions (if applicable by design) and have sell prices, amounts, and calculated profits.
    *   No orphaned records or inconsistent states (e.g., a sell without a corresponding buy if that's not allowed by the logic).
*   **Verification:** SQL queries on the database.

## VI. General Kubernetes Resource Behavior

### 1. Pod Restarts

*   **Test:** Manually delete a `trading-service` pod.
*   **Expected Outcome:** Kubernetes Deployment should automatically create a new pod to maintain the desired replica count. The new pod should start, load config, and become healthy.
*   **Verification:** `kubectl get pods`, `kubectl describe pod <new-pod-name>`, `kubectl logs <new-pod-name>`.

### 2. Job Cleanup (if configured)

*   **Test:** Observe completed or failed jobs.
*   **Expected Outcome:** If a `ttlSecondsAfterFinished` is configured for Jobs (either directly in TaskManager logic or a global policy), jobs and their pods should be automatically cleaned up after they finish.
*   **Verification:** `kubectl get jobs`, `kubectl get pods -l job-name=<job-name>`. Check if they disappear after the TTL. (Note: This might not be configured by default).

This manual testing guide provides a starting point. Specific test cases should be expanded based on the detailed requirements and implemented functionalities of the algorithms and error handling.
