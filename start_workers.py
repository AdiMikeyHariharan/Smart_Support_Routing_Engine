import os
import subprocess
import multiprocessing
import signal
import sys
import time

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Path to rq inside virtual environment
RQ_EXECUTABLE = os.path.join(PROJECT_ROOT, "env", "bin", "rq")

# Detect CPU cores
NUM_WORKERS = multiprocessing.cpu_count()

print(f"\nDetected {NUM_WORKERS} CPU cores")
print("Starting RQ workers...\n")

# ==========================================================
# Kill existing workers (avoid duplicates)
# ==========================================================

print("Stopping existing RQ workers (if any)...")
os.system("pkill -f 'rq worker'")
time.sleep(2)

# ==========================================================
# Start workers
# ==========================================================

worker_processes = []

for i in range(1, NUM_WORKERS + 1):
    worker_name = f"worker_{i}"
    log_file_path = os.path.join(LOG_DIR, f"{worker_name}.log")

    print(f"Starting {worker_name} → logging to {log_file_path}")

    log_file = open(log_file_path, "a", buffering=1)

    process = subprocess.Popen(
        [
            RQ_EXECUTABLE,
            "worker",
            "--with-scheduler",
            "--name",
            worker_name,
            "default",
        ],
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # 🔥 critical for detach
    )

    worker_processes.append(process)

print("\nAll workers started successfully.\n")

# ==========================================================
# Optional: Keep script alive (recommended for dev)
# ==========================================================

try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\nShutting down workers...")
    os.system("pkill -f 'rq worker'")
    sys.exit(0)