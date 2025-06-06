import uuid
import json
import logging
from kubernetes import client, config

# Configure basic logging for the TaskManager
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a constant for the namespace, can be made configurable
KUBE_NAMESPACE = "default"
# Define a constant for the Docker image, should be configurable
TASK_DOCKER_IMAGE = "your-docker-registry/trading-task:latest" # Placeholder
JOB_LABEL_SELECTOR = "app=crypto-trading-task,managed-by=task-manager"

class TaskManager:
    def __init__(self):
        self.tasks = {} # Stores task_id: {"job_name": job_name, "status": "pending/running/etc."}
        try:
            config.load_incluster_config()
            logging.info("Loaded in-cluster Kubernetes configuration.")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logging.info("Loaded local Kubernetes configuration.")
            except config.ConfigException as e:
                logging.error(f"Could not configure Kubernetes client: {e}")
                raise RuntimeError("Failed to configure Kubernetes client") from e

        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()
        logging.info("Kubernetes API clients initialized.")

    def _generate_job_name(self, task_id_str):
        # K8s names must be DNS-1123 compliant (lowercase alphanumeric, '-', '.')
        return f"trading-task-{task_id_str.lower().replace('_', '-')}"

    def create_task(self, config_obj):
        task_id = uuid.uuid4()
        task_id_str = str(task_id)
        job_name = self._generate_job_name(task_id_str)

        if not isinstance(config_obj, dict):
            logging.error("config_obj must be a dictionary.")
            return None

        container_config = config_obj.copy()
        container_config["task_id"] = task_id_str

        try:
            task_config_json = json.dumps(container_config)
        except TypeError as e:
            logging.error(f"Failed to serialize config_obj to JSON for task {task_id_str}: {e}")
            return None

        container = client.V1Container(
            name=f"trading-task-container-{task_id_str}",
            image=TASK_DOCKER_IMAGE,
            env=[client.V1EnvVar(name="TASK_CONFIG_JSON", value=task_config_json)],
        )

        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            containers=[container]
        )

        job_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": "crypto-trading-task", "task-id": task_id_str, "managed-by": "task-manager"}),
            spec=pod_spec
        )

        job_body = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name, labels={"app": "crypto-trading-task", "task-id": task_id_str, "managed-by": "task-manager"}),
            spec=client.V1JobSpec(
                template=job_template,
                backoff_limit=2,
            )
        )

        try:
            self.batch_v1.create_namespaced_job(body=job_body, namespace=KUBE_NAMESPACE)
            self.tasks[task_id_str] = {"job_name": job_name, "status": "pending"}
            logging.info(f"TaskManager: Created Kubernetes Job {job_name} for task {task_id_str}")
            return task_id_str
        except client.ApiException as e:
            logging.error(f"TaskManager: Error creating Kubernetes Job {job_name}: {e}")
            return None

    def stop_task(self, task_id_str):
        task_info = self.tasks.get(task_id_str)
        if not task_info:
            logging.warning(f"TaskManager: Task {task_id_str} not found for stopping.")
            return False

        job_name = task_info["job_name"]
        logging.info(f"TaskManager: Attempting to stop (delete) Kubernetes Job {job_name} for task {task_id_str}")

        try:
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=KUBE_NAMESPACE,
                body=client.V1DeleteOptions(propagation_policy='Background')
            )
            task_info["status"] = "stopped"
            logging.info(f"TaskManager: Successfully initiated deletion for Job {job_name}.")
            return True
        except client.ApiException as e:
            if e.status == 404:
                logging.warning(f"TaskManager: Job {job_name} not found for deletion (already deleted?).")
                task_info["status"] = "stopped"
                return True
            logging.error(f"TaskManager: Error deleting Kubernetes Job {job_name}: {e}")
            return False

    def get_task_status(self, task_id_str):
        task_info = self.tasks.get(task_id_str)
        if not task_info:
            logging.warning(f"TaskManager: Task {task_id_str} not found in local cache for status check.")
            return None

        job_name = task_info["job_name"]
        try:
            job_status_obj = self.batch_v1.read_namespaced_job_status(name=job_name, namespace=KUBE_NAMESPACE)

            if job_status_obj.status.succeeded and job_status_obj.status.succeeded > 0:
                task_info["status"] = "completed"
            elif job_status_obj.status.failed and job_status_obj.status.failed > 0:
                task_info["status"] = "failed"
            elif job_status_obj.status.active and job_status_obj.status.active > 0:
                task_info["status"] = "running"
            else:
                task_info["status"] = "pending"

            return task_info["status"]

        except client.ApiException as e:
            if e.status == 404:
                logging.warning(f"TaskManager: Job {job_name} for task {task_id_str} not found on Kubernetes. Marking as unknown/failed.")
                task_info["status"] = "failed"
                return task_info["status"]
            logging.error(f"TaskManager: Error getting status for Job {job_name}: {e}")
            return task_info.get("status", "unknown")

    def get_task_results(self, task_id_str):
        task_info = self.tasks.get(task_id_str)
        if not task_info:
            logging.warning(f"TaskManager: Task {task_id_str} not found for results.")
            return None

        job_name = task_info["job_name"]
        current_status = self.get_task_status(task_id_str)

        if current_status not in ["completed", "failed"]:
            logging.info(f"TaskManager: Task {task_id_str} (Job {job_name}) is not in a terminal state ({current_status}). Logs might be incomplete or unavailable.")

        try:
            pod_label_selector = f"job-name={job_name}"
            pods = self.core_v1.list_namespaced_pod(namespace=KUBE_NAMESPACE, label_selector=pod_label_selector)

            if not pods.items:
                logging.warning(f"TaskManager: No pods found for Job {job_name} (task {task_id_str}). Cannot fetch logs.")
                return None

            all_logs = []
            for pod_item in pods.items:
                pod_name = pod_item.metadata.name
                try:
                    log_message_prefix = f"Logs from pod {pod_name}"
                    if pod_item.status.phase in ["Succeeded", "Failed"]:
                        logging.info(f"TaskManager: Fetching logs for pod {pod_name} of Job {job_name} (task {task_id_str}).")
                        pod_logs = self.core_v1.read_namespaced_pod_log(name=pod_name, namespace=KUBE_NAMESPACE)
                        all_logs.append(log_message_prefix + ":\n" + pod_logs)
                    elif pod_item.status.phase == "Running" and current_status not in ["completed", "failed"]:
                        pod_logs = self.core_v1.read_namespaced_pod_log(name=pod_name, namespace=KUBE_NAMESPACE)
                        all_logs.append(log_message_prefix + " (task may be incomplete):\n" + pod_logs)
                    else:
                        logging.info(f"TaskManager: Pod {pod_name} for Job {job_name} is in phase {pod_item.status.phase}. Skipping log retrieval for this pod for now.")
                except client.ApiException as e_pod:
                    logging.error(f"TaskManager: Error fetching logs for pod {pod_name} (Job {job_name}): {e_pod}")
                    all_logs.append(f"Error fetching logs for pod {pod_name}: {str(e_pod)}")

            return "\n---\n".join(all_logs) if all_logs else "No logs found or task not in a state to fetch logs."

        except client.ApiException as e:
            logging.error(f"TaskManager: Error finding pods or fetching logs for Job {job_name}: {e}")
            return f"Error retrieving results: {str(e)}"

    def list_tasks(self):
        active_k8s_jobs = {}
        try:
            job_list = self.batch_v1.list_namespaced_job(namespace=KUBE_NAMESPACE, label_selector=JOB_LABEL_SELECTOR)
            for job in job_list.items:
                task_id_label = job.metadata.labels.get("task-id")
                if task_id_label:
                    active_k8s_jobs[task_id_label] = job.metadata.name
        except client.ApiException as e:
            logging.error(f"TaskManager: Error listing Kubernetes jobs: {e}")
            # Fallback to local cache if K8s API fails
            return {task_id: info.get("status", "unknown") for task_id, info in self.tasks.items()}

        # Sync local cache with Kubernetes state
        current_statuses = {}

        # Tasks currently in local cache: update their status or mark for removal
        tasks_to_remove_from_local_cache = []
        for task_id_str, task_info in list(self.tasks.items()):
            if task_id_str in active_k8s_jobs:
                # Job still exists in K8s, update local job name if it somehow changed (unlikely for same task_id)
                if task_info["job_name"] != active_k8s_jobs[task_id_str]:
                    logging.warning(f"Job name mismatch for task {task_id_str}: local '{task_info['job_name']}', k8s '{active_k8s_jobs[task_id_str]}'. Updating local.")
                    self.tasks[task_id_str]["job_name"] = active_k8s_jobs[task_id_str]
                current_statuses[task_id_str] = self.get_task_status(task_id_str) # Query fresh status
            else:
                # Job not found in K8s active jobs list (filtered by label)
                logging.info(f"Task {task_id_str} (Job {task_info['job_name']}) not found in Kubernetes active jobs. Marking as completed/failed or removing from cache.")
                # Preserve last known status if it was terminal, otherwise mark unknown
                if task_info.get("status") not in ["completed", "failed", "stopped"]:
                    self.tasks[task_id_str]["status"] = "unknown"
                current_statuses[task_id_str] = self.tasks[task_id_str]["status"]
                # If task is in a terminal state and not on K8s, it's safe to remove from local active cache
                if self.tasks[task_id_str]["status"] in ["completed", "failed", "stopped", "unknown"]:
                     tasks_to_remove_from_local_cache.append(task_id_str)

        # Add new jobs found in K8s that are not in local cache (e.g. if TaskManager restarted)
        for task_id_label, job_name_k8s in active_k8s_jobs.items():
            if task_id_label not in self.tasks:
                logging.info(f"Found job {job_name_k8s} for task {task_id_label} in Kubernetes not in local cache. Adding.")
                self.tasks[task_id_label] = {"job_name": job_name_k8s, "status": "unknown"} # Initial status unknown
                current_statuses[task_id_label] = self.get_task_status(task_id_label) # Query fresh status

        # Perform removal from local cache
        for task_id_to_remove in tasks_to_remove_from_local_cache:
            logging.info(f"Removing task {task_id_to_remove} from local cache as it's terminal and not listed in active K8s jobs.")
            if task_id_to_remove in self.tasks:
                del self.tasks[task_id_to_remove]
            # Also remove from current_statuses if it was added there and then deleted from self.tasks
            if task_id_to_remove in current_statuses and self.tasks.get(task_id_to_remove) is None:
                 del current_statuses[task_id_to_remove]

        # Return the status of tasks currently tracked (or recently found) by the TaskManager
        return {tid: self.tasks[tid]["status"] for tid in self.tasks if tid in self.tasks}


    def cleanup_completed_tasks(self):
        logging.info("TaskManager: Running cleanup of completed/failed tasks from internal cache.")
        # This method primarily cleans up the self.tasks dictionary.
        # K8s Job TTL controller or manual deletion is responsible for actual K8s job cleanup.
        tasks_to_remove = []
        for task_id, task_info in list(self.tasks.items()): # Iterate over a copy for safe deletion
            current_status = self.get_task_status(task_id) # This will also update task_info["status"]

            # If status indicates a terminal state (completed, failed)
            if current_status in ["completed", "failed"]:
                logging.info(f"Task {task_id} (Job {task_info['job_name']}) is in terminal state '{current_status}'. Verifying if K8s Job still exists before removing from local cache.")
                try:
                    # Check if the job still exists. If not (404), it's safe to remove from cache.
                    # If it exists, it might be waiting for TTL or manual cleanup.
                    # We remove from local cache because it's "done" from TaskManager's perspective.
                    self.batch_v1.read_namespaced_job_status(name=task_info["job_name"], namespace=KUBE_NAMESPACE)
                    logging.info(f"Job {task_info['job_name']} still exists on K8s. Task {task_id} will be removed from active tracking.")
                    tasks_to_remove.append(task_id)
                except client.ApiException as e:
                    if e.status == 404:
                        logging.info(f"Job {task_info['job_name']} for task {task_id} no longer exists on Kubernetes. Removing from cache.")
                        tasks_to_remove.append(task_id)
                    else:
                        # Log error but don't remove if we can't confirm its K8s status
                        logging.error(f"Error checking job {task_info['job_name']} during cleanup: {e}. Task {task_id} will not be removed from cache yet.")
            elif current_status == "stopped": # Explicitly stopped tasks
                 logging.info(f"Task {task_id} (Job {task_info['job_name']}) was stopped. Removing from active tracking.")
                 tasks_to_remove.append(task_id)
            elif current_status == "unknown": # If a task becomes unknown (e.g. K8s API issue during status check and job not found)
                logging.info(f"Task {task_id} (Job {task_info['job_name']}) has status 'unknown'. Removing from active tracking as its state is indeterminate from K8s.")
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            if task_id in self.tasks:
                del self.tasks[task_id]
        logging.info(f"TaskManager: Cleanup finished. Removed {len(tasks_to_remove)} tasks from internal active tracking.")


if __name__ == '__main__':
    # This block is for standalone testing or demonstration.
    # It requires a running Kubernetes cluster and proper configuration
    # (e.g., kubeconfig file or in-cluster service account).
    # Also, the Docker image TASK_DOCKER_IMAGE must exist and be accessible to the cluster.

    # --- Robust Logging Setup for __main__ ---
    log_level = logging.DEBUG
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Get the root logger
    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate messages if this script is re-run in some environments
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Configure new handlers
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout) # Ensure logs go to stdout

    # Reduce verbosity of Kubernetes client and urllib3 libraries
    logging.getLogger("kubernetes").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # --- End of Logging Setup ---

    main_logger = logging.getLogger("TaskManagerMain") # Use a specific logger for this main block
    main_logger.info("TaskManager Kubernetes example started.")

    try:
        manager = TaskManager()
        main_logger.info("TaskManager initialized.")

        # Example task configuration (ensure this matches what run_task_in_container.py expects)
        sample_config1 = {
            "currency": "BTC/USD",
            "exchange": "binance_simulation", # Assuming simulation mode for testing
            "strategy": "moving_average_crossover",
            "interval": "1m", # Use a short interval for faster completion in test
            "transaction_amt": 10, # Small amount
            "delay_secs": 5, # Short delay for the trading loop
            "connection_type": "simulation",
            "dir_path": "/app/config/sample_data/simu_data_btc_1m.csv", # Path inside the container
            "algo_config": { "short_window": 2, "long_window": 4 }, # Simplified for quick test
            "paper_trade": True, # Ensure it runs in paper trade mode
            # Ensure task_id will be added by create_task
        }

        main_logger.info(f"\nAttempting to create task 1 with config: {sample_config1}")
        task_id1 = manager.create_task(sample_config1)

        if task_id1:
            main_logger.info(f"Task 1 ({task_id1}) created. Initial status: {manager.get_task_status(task_id1)}")

            main_logger.info("Waiting a bit for K8s Job/Pod to potentially start (e.g., 20-30 seconds)...")
            import time
            time.sleep(30) # Adjust as needed based on cluster speed and image pull times

            main_logger.info(f"Status for task {task_id1} after delay: {manager.get_task_status(task_id1)}")

            main_logger.info("\nListing tasks from TaskManager:")
            main_logger.info(manager.list_tasks())

            main_logger.info(f"\nPolling status for task {task_id1} until completion/failure (max ~2 minutes)...")
            for i in range(24): # Poll for up to 120 seconds (24 * 5s)
                status = manager.get_task_status(task_id1)
                main_logger.info(f"[{i*5}s] Task {task_id1} status: {status}")
                if status in ["completed", "failed"]:
                    main_logger.info(f"Task {task_id1} reached terminal state: {status}")
                    break
                time.sleep(5)
            else: # Loop completed without break
                main_logger.warning(f"Task {task_id1} did not reach a terminal state within the polling period.")

            final_status_t1 = manager.get_task_status(task_id1)
            main_logger.info(f"\nFinal determined status for task {task_id1}: {final_status_t1}")

            main_logger.info(f"Attempting to retrieve results/logs for task {task_id1}:")
            results_t1 = manager.get_task_results(task_id1)
            if results_t1:
                # Ensure results_t1 is a string before attempting string operations
                results_str = str(results_t1)
                main_logger.info("Results/Logs (first 1000 chars):\n" + (results_str[:1000] + "..." if len(results_str) > 1000 else results_str))
            else:
                main_logger.info("No results/logs retrieved or task did not produce output that could be fetched.")

            # Example of stopping a task (optional, uncomment to test)
            # if final_status_t1 == "running" or final_status_t1 == "pending":
            #     main_logger.info(f"\nAttempting to stop task {task_id1}...")
            #     if manager.stop_task(task_id1):
            #         main_logger.info(f"Stop command issued for task {task_id1}. Status: {manager.get_task_status(task_id1)}")
            #         time.sleep(10) # Give K8s time to process deletion
            #         main_logger.info(f"Status after stop and delay: {manager.get_task_status(task_id1)}")
            #     else:
            #         main_logger.error(f"Failed to issue stop command for task {task_id1}.")

        else:
            main_logger.error("Failed to create task 1.")

        main_logger.info("\nRunning cleanup_completed_tasks...")
        manager.cleanup_completed_tasks()
        main_logger.info("\nList of tasks in TaskManager after cleanup:")
        main_logger.info(manager.list_tasks())

    except RuntimeError as e:
        main_logger.error(f"Runtime error during TaskManager initialization or operation: {e}", exc_info=True)
    except client.ApiException as e:
        main_logger.error(f"Kubernetes API Exception: {e.reason} (Status: {e.status}, Body: {e.body})", exc_info=True)
    except Exception as e_main:
        main_logger.error(f"An unexpected error occurred in the main example execution: {e_main}", exc_info=True)

    main_logger.info("\nTaskManager Kubernetes example finished.")
