import uuid
from .task import Task # Assuming task.py is in the same directory

class TaskManager:
    def __init__(self):
        self.tasks = {} # Dictionary to store task_id: Task object

    def create_task(self, config_obj):
        """
        Creates a new task, starts it, and stores it.
        Returns the task_id.
        """
        try:
            task = Task(config_obj)
            if task.start(): # Start the task
                self.tasks[task.task_id] = task
                # print(f"TaskManager: Created and started task {task.task_id}") # Placeholder
                return task.task_id
            else:
                # print(f"TaskManager: Failed to start task with config: {config_obj}") # Placeholder
                # Task status should reflect failure if start() returned False
                return None
        except Exception as e:
            # print(f"TaskManager: Error creating task: {e}") # Placeholder
            return None

    def stop_task(self, task_id):
        """
        Stops a specific task.
        Returns True if the task was found and stop signal was sent, False otherwise.
        """
        task = self.tasks.get(task_id)
        if task:
            # print(f"TaskManager: Stopping task {task_id}") # Placeholder
            return task.stop()
        # print(f"TaskManager: Task {task_id} not found for stopping.") # Placeholder
        return False

    def get_task_status(self, task_id):
        """
        Gets the status of a specific task.
        Returns the status string or None if task not found.
        """
        task = self.tasks.get(task_id)
        if task:
            return task.get_status()
        # print(f"TaskManager: Task {task_id} not found for status check.") # Placeholder
        return None

    def get_task_results(self, task_id):
        """
        Gets the results of a specific task.
        Returns the results or None if task not found.
        """
        task = self.tasks.get(task_id)
        if task:
            return task.get_results()
        # print(f"TaskManager: Task {task_id} not found for results.") # Placeholder
        return None

    def list_tasks(self):
        """
        Lists all tasks with their current status.
        Returns a dictionary of task_id: status.
        """
        return {task_id: task.get_status() for task_id, task in self.tasks.items()}

    def cleanup_completed_tasks(self):
        """
        Removes tasks that are in a final state (completed, stopped, failed)
        from the active tracking. This is important to prevent the self.tasks
        dictionary from growing indefinitely.
        Actual results/logs should be persisted elsewhere if needed long-term.
        """
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            status = task.get_status()
            if status in ["completed", "stopped", "failed"]:
                # Ensure process is joined before removing
                if task.process and task.process.is_alive():
                    task.process.join(timeout=1) # Give it a moment to close
                if not task.process or not task.process.is_alive(): # Check again
                    tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            # print(f"TaskManager: Removing task {task_id} from active tracking.") # Placeholder
            del self.tasks[task_id]


if __name__ == '__main__':
    # Example Usage (for testing purposes)
    import multiprocessing # For Event().wait()

    manager = TaskManager()
    print("TaskManager created.")

    # Create and start a couple of tasks
    print("\nCreating tasks...")
    config1 = {"id": 1, "detail": "Run strategy A on BTC/USD"}
    task_id1 = manager.create_task(config1)
    if task_id1:
        print(f"Task 1 ({task_id1}) created. Status: {manager.get_task_status(task_id1)}")
    else:
        print("Failed to create task 1")

    config2 = {"id": 2, "detail": "Run strategy B on ETH/EUR"}
    task_id2 = manager.create_task(config2)
    if task_id2:
        print(f"Task 2 ({task_id2}) created. Status: {manager.get_task_status(task_id2)}")
    else:
        print("Failed to create task 2")

    print("\nListing tasks after creation:")
    print(manager.list_tasks())

    print("\nWaiting for a few seconds...")
    multiprocessing.Event().wait(4) # Let tasks run

    print("\nListing tasks after some runtime:")
    print(manager.list_tasks())

    if task_id1:
        print(f"\nResults for task {task_id1}: {manager.get_task_results(task_id1)}")
        print(f"Status for task {task_id1}: {manager.get_task_status(task_id1)}")

    # Stop one task
    if task_id1:
        print(f"\nStopping task {task_id1}...")
        manager.stop_task(task_id1)
        print(f"Status for task {task_id1} after stop: {manager.get_task_status(task_id1)}")
        print(f"Results for task {task_id1} after stop: {manager.get_task_results(task_id1)}")


    print("\nListing tasks after stopping one:")
    print(manager.list_tasks())

    print("\nWaiting for other tasks to complete naturally...")
    # Wait for task2 to complete (max 15 seconds for the mock task)
    if task_id2:
        for _ in range(15):
            status_t2 = manager.get_task_status(task_id2)
            if status_t2 not in ["running", "starting"]:
                print(f"Task {task_id2} finished with status: {status_t2}")
                break
            multiprocessing.Event().wait(1)
        else:
            print(f"Task {task_id2} still running, stopping it.")
            manager.stop_task(task_id2)

    print("\nFinal list of tasks (before cleanup):")
    print(manager.list_tasks())

    if task_id2:
      print(f"\nFinal results for task {task_id2}: {manager.get_task_results(task_id2)}")


    print("\nRunning cleanup...")
    manager.cleanup_completed_tasks()
    print("\nList of tasks after cleanup:")
    print(manager.list_tasks())

    print("\nTaskManager example finished.")
