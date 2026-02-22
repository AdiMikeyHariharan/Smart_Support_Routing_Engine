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

RQ_EXECUTABLE = os.path.join(PROJECT_ROOT, "env", "bin", "rq")

# ----------------------------------------------------------
# IMPORTANT: Limit workers for ML workloads
# ----------------------------------------------------------
CPU_CORES = multiprocessing.cpu_count()

# Recommended for transformer inference on CPU
NUM_WORKERS = min(2, CPU_CORES)  # 🔥 Adjust if needed

print(f"\nDetected {CPU_CORES} CPU cores")
print(f"Starting {NUM_WORKERS} optimized RQ workers...\n")

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

    # Only FIRST worker gets scheduler
    if i == 1:
        args = [
            RQ_EXECUTABLE,
            "worker",
            "--with-scheduler",
            "--name",
            worker_name,
            "default",
        ]
    else:
        args = [
            RQ_EXECUTABLE,
            "worker",
            "--name",
            worker_name,
            "default",
        ]

    process = subprocess.Popen(
        args,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Detached
    )

    worker_processes.append(process)

print("\nAll workers started successfully.\n")

# ==========================================================
# Graceful Shutdown
# ==========================================================

def shutdown():
    print("\nShutting down workers gracefully...")
    for process in worker_processes:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            pass
    sys.exit(0)

try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    shutdown()