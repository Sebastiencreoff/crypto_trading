import multiprocessing
import uuid
from .trading import Trading # Uncommented
# from .config import Config # Config is now passed into Task

class Task:
    def __init__(self, config_obj):
        self.task_id = uuid.uuid4()
        self.config = config_obj
        self.process = None
        self.status = "pending"
        # For communication between processes (e.g., to send a stop signal)
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        # Potentially a queue for results or detailed status updates
        self.results_queue = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event() # Moved from _task_runner to be accessible by stop()

    def _task_runner(self, config_obj, child_conn, results_queue, task_id_str): # Added task_id_str
        """Internal method that will run in the separate process."""
        # self.status is managed by the main process based on process state / queue messages
        # No, _task_runner should update status via queue for more fine-grained states like "running".
        results_queue.put({"status_update": "running", "task_id": task_id_str, "message": f"Task {task_id_str} started."})

        stop_event_for_trading = multiprocessing.Event()
        trading_instance = Trading(config_obj=config_obj,
                                   task_id=task_id_str,
                                   stop_event=stop_event_for_trading,
                                   results_queue=results_queue)

        # Listener for stop signal from parent
        stop_listener_thread = None
        pipe_active = True

        def _listen_for_stop():
            nonlocal pipe_active
            try:
                while pipe_active:
                    if child_conn.poll(timeout=0.1): # Poll with timeout to allow loop to break
                        message = child_conn.recv()
                        if message == "stop":
                            results_queue.put({"status_update": "stopping", "task_id": task_id_str, "message": f"Task {task_id_str} received stop signal."})
                            stop_event_for_trading.set() # Signal Trading instance to stop
                            break
            except (EOFError, BrokenPipeError): # Pipe closed by parent
                results_queue.put({"status_update": "stopping", "task_id": task_id_str, "message": f"Task {task_id_str} pipe closed, initiating stop."})
                stop_event_for_trading.set() # Ensure trading instance stops if pipe breaks
            finally:
                pipe_active = False # Ensure listener exits

        try:
            stop_listener_thread = multiprocessing.Process(target=_listen_for_stop) # Run in separate process/thread
            stop_listener_thread.start()

            trading_instance.run() # This is the main blocking call

            # If trading_instance.run() completes without error, it's "completed"
            results_queue.put({"status_update": "completed", "task_id": task_id_str, "message": f"Task {task_id_str} completed normally."})

        except Exception as e:
            results_queue.put({"status_update": "failed", "task_id": task_id_str, "message": f"Task {task_id_str} failed: {str(e)}"})
        finally:
            pipe_active = False # Signal listener thread to exit
            if stop_listener_thread and stop_listener_thread.is_alive():
                 stop_listener_thread.join(timeout=1) # Wait for listener to exit
            if stop_listener_thread and stop_listener_thread.is_alive(): # If still alive
                 stop_listener_thread.terminate() # Force terminate
                 stop_listener_thread.join()

            if not stop_event_for_trading.is_set(): # Ensure it's set if trading_instance.run() exited unexpectedly
                stop_event_for_trading.set()
            child_conn.close()

    def start(self):
        if self.status == "pending":
            # task_id is a UUID object, convert to string for passing to process if needed,
            # though Trading class expects the UUID object directly now. Let's pass self.task_id
            self.process = multiprocessing.Process(
                target=self._task_runner,
                # Pass self.task_id (UUID) directly to _task_runner
                args=(self.config, self.child_conn, self.results_queue, self.task_id)
            )
            self.process.start()
            self.status = "starting" # Main process sets this, _task_runner will update to "running" via queue
            return True
        return False

    def stop(self):
        current_status = self.get_status() # Check current status before attempting to stop
        if current_status in ["running", "starting"]: # Only try to stop if it's logically running
            if self.process and self.process.is_alive():
                self.parent_conn.send("stop")
                # The _task_runner's listener thread will call stop_event.set() for the Trading instance.
                # Then Trading instance's run() loop will terminate.
                self.process.join(timeout=15) # Increased timeout for graceful shutdown
                if self.process.is_alive():
                    self.process.terminate() # Force terminate if graceful shutdown fails
                    self.process.join()
                self.status = "stopped" # Update status after attempting to stop
                return True
            else: # Process not alive, but status was running/starting
                self.status = "stopped" # Correct the status
                return False
        elif current_status in ["pending", "completed", "failed", "stopped"]:
            # If already in a terminal state or not yet started, no action needed or possible
            self.status = current_status # Ensure status is accurate if it was pending
            return False # Cannot stop something not running or already stopped/completed/failed
        return False


    def _update_status_from_queue(self):
        """Helper to process status messages from the results_queue."""
        # This should ideally be called by get_status or periodically by TaskManager
        # For now, it's a helper that can be integrated into get_status logic
        try:
            while True: # Process all available messages
                message = self.results_queue.get_nowait()
                if isinstance(message, dict) and "status_update" in message:
                    new_status = message["status_update"]
                    # Potentially log message["message"] here
                    # print(f"Task {self.task_id} status update from queue: {new_status} - {message.get('message')}")
                    # More complex status logic can be here if needed
                    # For example, don't override "failed" with "stopped" if failure happened first.
                    if self.status == "failed" and new_status == "stopped": # if already failed, don't overwrite
                        pass
                    else:
                        self.status = new_status
                else:
                    # This is a result message, not a status update, put it back for get_results
                    # This needs careful handling to avoid infinite loop if queue only has results
                    # For now, assume status_updates are distinct or handled by get_results
                    # A better approach: two queues, one for status, one for results.
                    # Or, TaskManager is responsible for polling results_queue and updating task status.
                    # For now, we'll just update self.status based on what we find.
                    pass # Let get_results handle non-status messages
        except multiprocessing.queues.Empty:
            pass # No new status messages

    def get_status(self):
        self._update_status_from_queue() # Check queue for latest status from _task_runner

        if self.process and self.process.is_alive():
            if self.status in ["pending", "starting"]: # If process is alive, it should be at least 'running'
                 # unless _task_runner hasn't sent first status update yet
                return "running" # Override if not updated by queue yet but process is live
            return self.status # Return status set by _task_runner via queue

        # Process is not alive, determine final status
        self._update_status_from_queue() # One last check

        if self.status in ["completed", "stopped", "failed"]:
            return self.status # Return final status set by _task_runner or stop()

        # If process died unexpectedly without _task_runner setting final status
        if self.process and self.process.exitcode is not None:
            if self.process.exitcode == 0:
                # If it exited cleanly and status isn't already terminal, mark completed
                if self.status not in ["completed", "stopped", "failed"]:
                    self.status = "completed"
            else: # Non-zero exit code implies failure
                if self.status not in ["completed", "stopped", "failed"]:
                    self.status = "failed"
            return self.status

        # Default to current status if process info is not definitive
        return self.status


    def get_results(self):
        # Note: _update_status_from_queue might consume status messages.
        # This assumes results_queue can contain mixed message types (dicts for status, others for results)
        # or that status messages are handled/filtered appropriately.
        # A cleaner design might use separate queues for status and results.

        # First, ensure status is up-to-date, as this might process some queue items
        self._update_status_from_queue()

        results = []
        try:
            while True: # Get all available actual results
                item = self.results_queue.get_nowait()
                if isinstance(item, dict) and "status_update" in item:
                    # This is a status message, which _update_status_from_queue should have handled.
                    # If we find one here, it means it was put after the last call. Re-evaluate status.
                    # For simplicity here, we'll ignore it as status should be primary concern of get_status.
                    # Or, if this task instance is the sole consumer, update status here too.
                    new_status = item["status_update"]
                    if self.status == "failed" and new_status == "stopped":
                        pass
                    else:
                        self.status = new_status
                else:
                    results.append(item) # Actual result
        except multiprocessing.queues.Empty:
            pass # No more items in the queue
        return results

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # This example usage will likely fail or behave differently due to changes in how
    # config (especially db_conn) is expected by the refactored Trading class.
    # It needs a proper Config object with a db_conn attribute.
    # For now, we'll comment out parts that would clearly fail without a real db setup.

    print("Creating task...")
    # Mock config_obj that Trading class expects.
    # Needs db_conn, currency, connection_type, connection_config, dir_path (if sim),
    # algo_config, transaction_amt, delay_secs.
    # This is a placeholder and won't fully work without a database connection.
    mock_db_conn = None # In a real scenario, this would be a database connection object.

    # It's better to test this with TaskManager and a proper Config setup.
    # The following example is simplified and focuses on Task mechanics.

    class MockConfig:
        def __init__(self, currency, delay=1, connection_type="simulation"):
            self.currency = currency
            self.delay_secs = delay # Expected by refactored Trading
            self.transaction_amt = 100 # Expected
            self.db_conn = mock_db_conn # Expected
            self.connection_type = connection_type
            # Provide dummy connection_config and dir_path for simulation
            self.connection_config = {} # Dummy
            self.dir_path = "."         # Dummy
            # Provide dummy algo_config
            self.algo_config = {}       # Dummy
            self.logger = logging.getLogger("MockConfigLogger")


    # task_config = {"currency_pair": "BTC/USD", "amount": 100} # Old style
    task_config_obj = MockConfig(currency="BTC/USD")
    t = Task(task_config_obj)
    print("Creating task...")
    task_config = {"currency_pair": "BTC/USD", "amount": 100}
    t = Task(task_config)
    print(f"Task ID: {t.task_id}, Status: {t.get_status()}")

    print("Starting task...")
    t.start()
    # Status might be 'starting' initially, then 'running' after _task_runner sends update
    multiprocessing.Event().wait(0.5) # Give it a moment to start and update status
    print(f"Task ID: {t.task_id}, Status: {t.get_status()}")


    print("Simulating run for a few seconds (Task is mock, won't do real trading)...")
    # The mock Trading class in this example doesn't have a long run loop by default
    # The original Trading class would run until data ends or stopped.
    # We're relying on the refactored Trading class's behavior.
    # For this test, the mock trading loop inside original task.py was 10s.
    # The refactored Trading class might finish quickly if not configured for long run.
    # Let's assume it runs for a bit or finishes.

    # Wait for the task to potentially finish or run for a bit
    for i in range(5): # Check status over 5 seconds
        s = t.get_status()
        r = t.get_results()
        print(f"After {i+1}s: Task ID: {t.task_id}, Status: {s}, Results: {r}")
        if s not in ["running", "starting"]:
            break
        multiprocessing.Event().wait(1)


    print("Stopping task (if it's still running)...")
    stopped = t.stop()
    print(f"Stop attempted: {stopped}. Task ID: {t.task_id}, Status: {t.get_status()}")

    print("Final results...")
    print(f"Results: {t.get_results()}")
    print(f"Task ID: {t.task_id}, Final Status: {t.get_status()}")


    print("\nTrying to start a new task to see if it completes (mock behavior)...")
    task_config_obj2 = MockConfig(currency="ETH/USD", delay=1)
    t2 = Task(task_config_obj2)
    t2.start()
    print(f"Task ID: {t2.task_id}, Status: {t2.get_status()}")

    # Wait for t2 to complete or run for its duration
    # The internal loop of refactored Trading depends on data or stop signal.
    # This test depends on how quickly the (simulated) data source in Trading would exhaust.
    # For this example, let's assume it might run for up to its internal loop (e.g. 10s if that was kept)
    # or until the simulation data ends.
    for i in range(12): # Max wait for 12 seconds
        status_t2 = t2.get_status()
        results_t2 = t2.get_results() # Get results to clear queue for next status update
        print(f"Task ID: {t2.task_id}, Status: {status_t2}, Results: {results_t2}")
        if status_t2 not in ["running", "starting"]:
            break
        multiprocessing.Event().wait(1)

    print(f"Task {t2.task_id} final status: {t2.get_status()}")
    print(f"Task {t2.task_id} final results: {t2.get_results()}") # Get any last results

    print("\nAll tasks done (example finished).")
