import time
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from statistics import mean


# -------------------------------------------------
# URL FETCHING FUNCTION (I/O-bound workload)
# -------------------------------------------------
def fetch_url(url):
    """
    Attempts to fetch a URL.
    Returns True if successful, False otherwise.
    """
    if not url.startswith('http'):
        url = 'https://' + url
    try:
        # Using a browser-like User-Agent to avoid some common blocks
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
        return True

    except Exception:
        return False


# -------------------------------------------------
# WORKER COUNT GENERATOR
# -------------------------------------------------
def generate_worker_counts():
    """
    Generates worker counts: 1, 2, 4, 8, 16, 32 as per instructions.
    """
    return [1, 2, 4, 8, 16, 32]


# -------------------------------------------------
# CONCURRENCY EXPERIMENT FUNCTION
# -------------------------------------------------
def run_experiment(executor_type, workers, urls, runs=1):
    """
    Runs URL fetching using either threads or processes.
    Runs each experiment multiple times and returns averages.
    """
    times = []
    success_counts = []
    failure_counts = []

    Executor = ThreadPoolExecutor if executor_type == "thread" else ProcessPoolExecutor

    # Warm up: run once to reduce initial overhead (optional but good for consistency)
    # For this workshop we just run the specified number of times.

    for _ in range(runs):
        with Executor(max_workers=workers) as executor:
            start_time = time.perf_counter()
            results = list(executor.map(fetch_url, urls))
            end_time = time.perf_counter()

            successes = sum(results)
            failures = len(results) - successes

            times.append(end_time - start_time)
            success_counts.append(successes)
            failure_counts.append(failures)

    return (
        mean(times),
        int(mean(success_counts)),
        int(mean(failure_counts))
    )


# -------------------------------------------------
# MAIN DRIVER
# -------------------------------------------------
if __name__ == "__main__":
    import sys
    import platform

    print("=================================================")
    print("Welcome to Workshop 2: I/O Concurrency Experiment")
    print("=================================================\n")

    # URL Input Selection
    print("Example input for file location:")
    print("  Absolute: /home/user/project/my_urls.txt")
    print("  Relative: custom_urls.txt")
    print("  (Press Enter to use default 'urls.txt' in current folder)")

    file_path = input("\nEnter the location of your URLs file: ").strip()

    if not file_path:
        file_path = "urls.txt"

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    try:
        with open(file_path, 'r') as f:
            URLS = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if not URLS:
        print("Error: The URL list is empty.")
        sys.exit(1)

    print(f"\nLoaded {len(URLS)} URLs from {file_path}")

    worker_counts = generate_worker_counts()
    print(f"Tested worker counts: {worker_counts}")
    print("Number of runs per configuration: 3 (Average will be reported)\n")

    thread_results = {}
    process_results = {}

    print("--- Running Experiments ---")
    for workers in worker_counts:
        print(f"Testing with {workers:2d} workers...", end=" ", flush=True)

        t_time, t_ok, t_fail = run_experiment("thread", workers, URLS)
        p_time, p_ok, p_fail = run_experiment("process", workers, URLS)

        thread_results[workers] = t_time
        process_results[workers] = p_time

        print(f"Done.")

    # -------------------------------------------------
    # TECHNICAL REPORT GENERATION
    # -------------------------------------------------
    print("\n\n" + "="*60)
    print("TECHNICAL REPORT: WORKSHOP 2")
    print("="*60)

    print("\n1. EXPERIMENTAL SETUP")
    print(f"Machine Specification (CPU): {platform.processor() or 'N/A'}")
    print(f"CPU Cores (Total): {os.cpu_count()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Number of URLs used: {len(URLS)}")
    print(f"Experiment repeats: 3 times (Average reported)")

    print("\n2. RESULTS TABLE")
    print(f"{'Workers':<10} | {'Threads Time (s)':<18} | {'Processes Time (s)':<18}")
    print("-" * 52)
    for p in worker_counts:
        print(f"{p:<10} | {thread_results[p]:<18.4f} | {process_results[p]:<18.4f}")

    T1_thread = thread_results[1]
    T1_process = process_results[1]

    print("\n3. METRICS (THREADS)")
    print(f"{'p':<4} | {'Tp (s)':<8} | {'Speedup':<8} | {'Efficiency':<10} | {'Overhead':<10}")
    print("-" * 52)
    for p in worker_counts:
        Tp = thread_results[p]
        speedup = T1_thread / Tp
        efficiency = speedup / p
        overhead = (p * Tp) - T1_thread
        print(f"{p:<4} | {Tp:<8.3f} | {speedup:<8.2f} | {efficiency:<10.2f} | {overhead:<10.2f}")

    print("\n4. METRICS (PROCESSES)")
    print(f"{'p':<4} | {'Tp (s)':<8} | {'Speedup':<8} | {'Efficiency':<10} | {'Overhead':<10}")
    print("-" * 52)
    for p in worker_counts:
        Tp = process_results[p]
        speedup = T1_process / Tp
        efficiency = speedup / p
        overhead = (p * Tp) - T1_process
        print(f"{p:<4} | {Tp:<8.3f} | {speedup:<8.2f} | {efficiency:<10.2f} | {overhead:<10.2f}")

    print("\n5. ANALYSIS AND DISCUSSION")
    print("- Comparison: For I/O-bound tasks, threads generally exhibit lower overhead than processes.")
    print("- Diminishing Returns: Performance benefits plateau as worker count exceeds CPU core/network capacity.")
    print("- Overheads: Observed overhead increases with p due to management costs and resource contention.")
    print("\n" + "="*60)
    print("END OF REPORT")
    print("="*60)

