import threading
import time
import os

# --- 1. Sequential Implementation ---
def calculate_sequential(start, end, step):
    # sum() with a generator is highly optimized in C
    return sum(i * i for i in range(start, end, step))

# --- 2. Parallel Implementation ---
shared_sum = 0
mutex = threading.Lock()

def parallel_worker(start, end, step):
    global shared_sum
    
    # Calculate local sum first to minimize time spent holding the lock
    local_sum = sum(i * i for i in range(start, end, step))
    
    # Thread Synchronization: Acquiring the mutex before modifying shared data
    # This avoids race conditions where multiple threads might try to write to
    # shared_sum simultaneously, corrupting the final value.
    with mutex:
        shared_sum += local_sum

def calculate_parallel(start, end, step, num_threads):
    global shared_sum
    shared_sum = 0 
    
    # Mathematical range generation to accurately divide workloads
    r = range(start, end, step)
    total_elements = len(r)
    if total_elements == 0: return 0
    
    chunk_size = total_elements // num_threads
    threads = []
    
    for i in range(num_threads):
        # Slice the range for exact distribution
        # This handles uneven divisions cleanly
        t_start = r.start + (i * chunk_size * r.step)
        if i == num_threads - 1:
            t_end = r.stop
        else:
            t_end = t_start + (chunk_size * r.step)
            
        t = threading.Thread(target=parallel_worker, args=(t_start, t_end, r.step))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
        
    return shared_sum

# --- 3. Experiment and Benchmarks ---
if __name__ == "__main__":
    START, END, STEP = 1, 10_000_000, 1
    MAX_THREADS = os.cpu_count() or 8
    
    print("--- Optimized Threading Experiment ---")
    
    # Baseline Sequential
    t0 = time.time()
    res_seq = calculate_sequential(START, END, STEP)
    t_seq = time.time() - t0
    
    print(f"Sequential Execution Time: {t_seq:.4f} seconds (Result: {res_seq})")
    print("-" * 75)
    print(f"{'Threads':<10} | {'Time (s)':<10} | {'Speedup':<10} | {'Efficiency':<12} | {'Overhead (s)'}")
    print("-" * 75)
    
    results = []
    thread_counts = [2**i for i in range(1, 7) if 2**i <= MAX_THREADS * 2]
    
    for n in thread_counts:
        t0 = time.time()
        res_par = calculate_parallel(START, END, STEP, n)
        t_par = time.time() - t0
        
        assert res_par == res_seq, "Parallel calculation error! Results do not match."
        
        speedup = t_seq / t_par
        efficiency = speedup / n
        overhead = t_par - (t_seq / n)
        
        results.append((n, t_par, speedup, efficiency, overhead))
        print(f"{n:<10} | {t_par:<10.4f} | {speedup:<10.4f} | {efficiency:<12.4f} | {overhead:.4f}")

    print("\n--- Analysis & Discussion ---")
    print("1. Speedup & Efficiency: ")
    print("   Notice the Speedup hovers around 1.0 (or slightly below). This is expected in Python.")
    print("   Due to the Global Interpreter Lock (GIL), only one thread executes Python bytecode at a time.")
    print("   Therefore, CPU-bound tasks like mathematical calculations see NO speedup from standard multithreading.")
    print("\n2. Threading Overhead: ")
    print("   As the number of threads increases, execution time might actually increase (speedup < 1).")
    print("   This is the 'overhead' of the operating system provisioning threads and context-switching.")
    print("\n3. Importance of Mutex Synchronization: ")
    print("   We calculate 'local_sum' outside the lock, and only use 'with mutex:' to update 'shared_sum'.")
    print("   If we locked the mutex on every step of the loop, performance would plummet drastically.")
    print("   Without the mutex entirely, concurrent writes to 'shared_sum' would cause race conditions, yielding incorrect results.")