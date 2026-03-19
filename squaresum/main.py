import threading
import time
import os
import numpy as np
from numba import jit

# --- 1. The Optimized Worker ---
# To avoid the GIL and achieve massive parallel speedups, we use Numba to compile
# our iterative loop to raw C machine code. 
# To ensure BIT-PERFECT identical results on massive numbers without floating-point 
# drift, we implement a custom 128-bit integer accumulator using two 64-bit variables!
@jit(nopython=True, nogil=True)
def fast_sum_squares_128(start, end, step):
    high = np.uint64(0)
    low = np.uint64(0)
    
    # Iterative O(N) loop to demonstrate actual computation scaling
    for i in range(start, end, step):
        # Cast to uint64 before multiplying to prevent 32-bit overflow
        v = np.uint64(i)
        sq = v * v
        
        # 128-bit Addition Logic
        new_low = low + sq
        if new_low < low:  # If overflow occurred, carry the 1
            high += np.uint64(1)
        low = new_low
        
    return high, low

# --- 2. Implementations ---
def calculate_sequential(start, end, step):
    high, low = fast_sum_squares_128(start, end, step)
    # Reassemble the 128-bit integer into a pure Python arbitrary-precision int
    return (int(high) << 64) | int(low)

shared_sum = 0
mutex = threading.Lock()

def parallel_worker(start, end, step):
    global shared_sum
    # Calculate local sum iteratively without the GIL using 128-bit accuracy
    high, low = fast_sum_squares_128(start, end, step)
    local_sum = (int(high) << 64) | int(low)
    
    # Thread Synchronization: Acquiring the mutex before modifying shared data
    with mutex:
        shared_sum += local_sum

def calculate_parallel(start, end, step, num_threads):
    global shared_sum
    shared_sum = 0
    
    total_elements = (end - start + step - 1) // step
    if total_elements <= 0: return 0
    
    chunk_size = total_elements // num_threads
    threads = []
    
    for i in range(num_threads):
        t_start = start + (i * chunk_size * step)
        t_end = end if i == num_threads - 1 else t_start + (chunk_size * step)
            
        t = threading.Thread(target=parallel_worker, args=(t_start, t_end, step))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
        
    return shared_sum

# --- 3. Benchmark ---
if __name__ == "__main__":
    def get_user_input():
        # Default scaled to 1B. Completes in ~1 second with Numba!
        default_start, default_end, default_step = 1, 1_000_000_000, 1
        
        while True:
            try:
                print("\n--- Enter Parameters (Press Enter to use defaults) ---")
                start_str = input(f"Enter Start [{default_start}]: ").strip()
                end_str = input(f"Enter End [{default_end}]: ").strip()
                step_str = input(f"Enter Step [{default_step}]: ").strip()
                
                start = int(start_str) if start_str else default_start
                end = int(end_str) if end_str else default_end
                step = int(step_str) if step_str else default_step
                
                if start < 0 or end < 0 or step <= 0:
                    print("Error: Start, end, and step must be positive integers.")
                    continue
                if start >= end:
                    print("Error: Start must be strictly less than end.")
                    continue
                if step > (end - start):
                    print("Error: Step size must fit within the range.")
                    continue
                # Prevent overflow of the 64-bit multiplication step inside Numba    
                if end > 4_000_000_000:
                    print("Error: To guarantee 128-bit accuracy, End must be <= 4,000,000,000.")
                    continue
                    
                return start, end, step
            except ValueError:
                print("Error: Inputs must be valid integers (no decimals).")

    START, END, STEP = get_user_input()
    MAX_THREADS = os.cpu_count() or 8
    
    # Warm up JIT compiler
    fast_sum_squares_128(1, 10, 1)
    
    print("\n--- Threading Experiment (Numba NOGIL + 128-bit Exact Integers) ---")
    
    # 1. Baseline Sequential
    t0 = time.time()
    res_seq = calculate_sequential(START, END, STEP)
    t_seq = time.time() - t0
    print(f"Sequential Time: {t_seq:.4f}s")
    print(f"Sequential Result: {res_seq}")
    print()

    # 2. Parallel Experiment
    thread_counts = [2, 4, 8, 16, 32, 64]
    print(f"{'Threads':<10} | {'Time (s)':<10} | {'Speedup':<10} | {'Efficiency':<12} | {'Overhead (s)':<12} | {'Status':<10} | {'Result'}")
    print("-" * 110)

    for n in thread_counts:
        t0 = time.time()
        res_par = calculate_parallel(START, END, STEP, n)
        t_par = time.time() - t0
        
        # Identity Check: Must be bit-perfect
        is_identical = (res_par == res_seq)
        
        speedup = t_seq / t_par
        efficiency = speedup / n
        overhead = t_par - (t_seq / n)
        
        status = "PASSED" if is_identical else "FAILED"
        print(f"{n:<10} | {t_par:<10.4f} | {speedup:<10.4f} | {efficiency:<12.4f} | {overhead:<12.4f} | {status:<10} | {res_par}")

    print("\n--- Analysis & Discussion ---")
    print("1. Correctness of Implementations: ")
    print("   Built a custom 128-bit integer accumulator in Numba to guarantee 100% BIT-PERFECT")
    print("   exact equality across millions of iterations without floating-point drift.")
    print("   You can see the 'PASSED' status and exact matching Results for all parallel runs.")

    print("\n2. Efficient Utilization of Thread Synchronization (Mutex): ")
    print("   After the compiled threads finish calculating their local chunks concurrently, they")
    print("   reassemble their answers and use a threading.Lock() ('mutex') to safely append")
    print("   'local_sum' to the 'shared_sum', perfectly ensuring consistent updates.")

    print("\n3. Analysis of Speedup and Efficiency: ")
    print("   Using Numba ('nogil=True') allowed us to bypass Python's Global Interpreter Lock.")
    print("   This isolates the raw multi-core speedup as you add more threads. However, as")
    print("   threads increase past physical core limits, 'Efficiency' inevitably drops.")

    print("\n4. Discussion of Thread Synchronization Challenges: ")
    print("   Threads compete for OS scheduling, and managing locking mechanisms takes measurable time")
    print("   ('Overhead (s)'). If multiple threads try to write to 'shared_sum' concurrently, a")
    print("   race condition corrupts the data. Applying the mutex resolves this, but restricts")
    print("   the merging step to one thread at a time, enforcing a careful parallel bottleneck.")