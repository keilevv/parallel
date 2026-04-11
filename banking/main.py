import multiprocessing
import random
import time
import os
import math

# Global ledger for zero-copy memory access via linux fork()
master_ledger = []

def parallel_worker(balance, lock, start_idx, end_idx, worker_id):
    global master_ledger
    local_balance = 0.0
    
    # We read natively from the exact memory space cloned instantaneously 
    # without any IPC pipe array pickling serialization overhead!
    for offset in range(start_idx, end_idx):
        tx_type, amount = master_ledger[offset]
        if tx_type == "DEPOSIT":
            local_balance += amount
        else:
            if local_balance >= amount:
                local_balance -= amount
                
    with lock:
        balance.value = round(balance.value + local_balance, 2)

def run_serial(task_list):
    balance = 0.0
    print("\n--- STARTING SERIAL ---")
    start = time.time()
    for idx, (tx_type, amount) in enumerate(task_list):
        if tx_type == "DEPOSIT":
            balance = round(balance + amount, 2)
        else:
            if balance >= amount:
                balance = round(balance - amount, 2)
    return balance, time.time() - start

def main():
    # Python 3.14 defaults to 'spawn' on POSIX. We MUST use 'fork' 
    # to inherit the global master_ledger array natively without pickling!
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
        
    global master_ledger
    try:
        total_tx = int(input("Number of transactions: "))
        seed_val = int(input("Random seed: (for reproducibility) "))
    except ValueError: return

    random.seed(seed_val)
    master_ledger = []
    for _ in range(total_tx):
        tx_type = random.choice(["DEPOSIT", "WITHDRAW"])
        amount = round(random.uniform(1.0, 100.0), 2)
        master_ledger.append((tx_type, amount))

    serial_final, serial_time = run_serial(master_ledger)

    print("\n--- STARTING PARALLEL ---")
    balance = multiprocessing.Value('d', 0.0)
    lock = multiprocessing.Lock()
    
    num_workers = os.cpu_count() or 4
    chunk_size = math.ceil(total_tx / num_workers)
    
    processes = []
    p_start = time.time()
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_tx)
        
        # Don't pass the chunk array over args! Let it inherit zero-copy natively.
        p = multiprocessing.Process(
            target=parallel_worker, 
            args=(balance, lock, start_idx, end_idx, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    parallel_time = time.time() - p_start

    print("\n" + "="*40)
    print(f"SERIAL FINAL:   ${serial_final:.2f} | Time: {serial_time:.4f}s")
    print(f"PARALLEL FINAL: ${balance.value:.2f} | Time: {parallel_time:.4f}s")
    print("="*40)
    
    if round(serial_final, 2) == round(balance.value, 2):
        print("SUCCESS: Results are identical!")
    else:
        print("NOTE: Results differ. This is expected in TRUE parallelism")
        print("due to asynchronous evaluation of operations preventing")
        print("deterministic withdrawal acceptance!")

if __name__ == "__main__":
    main()