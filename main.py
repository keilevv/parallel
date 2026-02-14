import time
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from statistics import mean
import random
import platform
import sys


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
        with urllib.request.urlopen(req, timeout=3) as response:
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
def run_experiment(executor_type, workers, urls):
    """
    Runs URL fetching using either threads or processes.
    Uses as_completed to handle results as they finish.
    """
    import concurrent.futures
    
    Executor = ThreadPoolExecutor if executor_type == "thread" else ProcessPoolExecutor
    total_urls = len(urls)
    
    # Shuffle URLs to avoid clustering slow ones (stragglers) at the end
    shuffled_urls = list(urls)
    random.shuffle(shuffled_urls)

    with Executor(max_workers=workers) as executor:
        start_time = time.perf_counter()
        
        # Submit all tasks
        futures = {executor.submit(fetch_url, url): url for url in shuffled_urls}
        
        results = []
        completed = 0
        total = len(futures)
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            
            # Active workers is total submitted minus those already completed
            # but capped by the pool size (workers)
            active = min(workers, total - completed + 1) if (total - completed + 1) > 0 else 0
            
            print(f"\r  [{executor_type.capitalize()} p={workers}] "
                  f"Progress: {completed:2d}/{total_urls} | "
                  f"Active Workers: {active:2d} ", end="", flush=True)
        
        end_time = time.perf_counter()
        print() # New line after progress

        successes = sum(results)
        execution_time = end_time - start_time

    return execution_time, successes, (total_urls - successes)


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
    print(f"Tested worker counts: {worker_counts}\n")

    thread_results = {}
    process_results = {}

    print("--- Running Experiments (with URL Shuffling) ---")
    for workers in worker_counts:
        t_time, t_ok, t_fail = run_experiment("thread", workers, URLS)
        p_time, p_ok, p_fail = run_experiment("process", workers, URLS)

        thread_results[workers] = t_time
        process_results[workers] = p_time

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
    print(f"Experiment repeats: 1 time")

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

    # -------------------------------------------------
    # OPTIMAL CONFIGURATION ANALYSIS
    # -------------------------------------------------
    all_configs = []
    for p in worker_counts:
        # Threads
        t_time = thread_results[p]
        t_speedup = T1_thread / t_time
        t_efficiency = t_speedup / p
        all_configs.append({
            "type": "Threads", "p": p, "time": t_time, 
            "speedup": t_speedup, "efficiency": t_efficiency
        })
        # Processes
        p_val = process_results[p]
        p_speedup = T1_process / p_val
        p_efficiency = p_speedup / p
        all_configs.append({
            "type": "Processes", "p": p, "time": p_val, 
            "speedup": p_speedup, "efficiency": p_efficiency
        })

    # Find optimal: Best speedup with Efficiency >= 0.5
    # If none meet the threshold (unlikely for I/O), pick highest efficiency.
    threshold = 0.5
    eligible_configs = [c for c in all_configs if c['efficiency'] >= threshold and c['p'] > 1]
    
    if eligible_configs:
        # Of those that are efficient enough, pick the fastest (highest speedup)
        optimal = sorted(eligible_configs, key=lambda x: x['speedup'], reverse=True)[0]
    else:
        # Fallback to the single best efficiency (likely p=2 or p=4)
        optimal = sorted(all_configs, key=lambda x: x['efficiency'], reverse=True)[0]

    print("\n6. OPTIMAL CONFIGURATION")
    print(f"Based on a balanced analysis of Speedup and Efficiency:")
    print(f"- Recommended Configuration: {optimal['type']} with {optimal['p']} workers")
    print(f"- Execution Time: {optimal['time']:.4f}s")
    print(f"- Speedup: {optimal['speedup']:.2f}x")
    print(f"- Efficiency: {optimal['efficiency']:.2f}")
    print(f"- Rationale: This setup provides the best performance gain while maintaining ")
    print(f"  resource utilization above {threshold*100:.0f}%. Higher worker counts were discarded ")
    print(f"  due to diminishing returns and excessive overhead.")

    print("\n" + "="*60)
    print("END OF REPORT")
    print("="*60)

