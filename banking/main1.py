import math, os, random, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit

# ── Kernels ───────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def _scan_chunk(types, amounts, start, end, opening):
    bal = opening
    for i in range(start, end):
        if types[i] == 1:
            bal += amounts[i]
        else:
            if bal >= amounts[i]:
                bal -= amounts[i]
    return bal

@njit(cache=True)
def run_serial_numba(types, amounts):
    return _scan_chunk(types, amounts, 0, len(types), 0.0)

# ── Parallel runner ───────────────────────────────────────────────────────────

def run_parallel(types, amounts, num_workers, pool):
    n = len(types)
    chunk = math.ceil(n / num_workers)
    starts = [i * chunk         for i in range(num_workers)]
    ends   = [min(s + chunk, n) for s in starts]

    # Pass 2: O(num_workers) serial prefix scan
    # Re-scans only num_workers chunks to thread real opening balances through.
    # This is cheap: e.g. 4 chunks * (10M/4 transactions) would be slow,
    # BUT we only need the closing balance of chunk i-1 to get opening of chunk i,
    # and we process them one at a time serially — total work = one full serial scan.
    # To make this sublinear we'd need chunk summaries from Pass 1 (see below).
    #
    # The trick: run Pass 1 and Pass 2 overlap — do Pass 1 in parallel while
    # Pass 2 processes the previous chunk's result as soon as it arrives.
    # Use as_completed so Pass 2 can start the moment chunk 0 finishes.

    from concurrent.futures import as_completed

    # Submit all Pass-1 futures immediately
    futures = {
        pool.submit(_scan_chunk, types, amounts, starts[i], ends[i], 0.0): i
        for i in range(num_workers)
    }

    pass1 = [None] * num_workers
    for fut in as_completed(futures):
        pass1[futures[fut]] = fut.result()

    # Pass 2: compute real openings in O(num_workers) re-scans of chunk boundaries
    # Each re-scan is one chunk — total = one serial scan worth of work.
    # We parallelise this differently: chunk i's opening only depends on chunk i-1,
    # so it's a strict sequential chain. Accept the cost: num_workers re-scans
    # are unavoidable for exact correctness.
    openings = [0.0] * num_workers
    for i in range(1, num_workers):
        openings[i] = _scan_chunk(
            types, amounts, starts[i-1], ends[i-1], openings[i-1]
        )

    # Pass 3: parallel re-scan with correct openings
    finals = list(pool.map(
        lambda i: _scan_chunk(types, amounts, starts[i], ends[i], openings[i]),
        range(num_workers)
    ))

    return finals[-1]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_arrays(ledger):
    types   = np.array([1 if t == "DEPOSIT" else 0 for t, _ in ledger], dtype=np.int8)
    amounts = np.array([a for _, a in ledger], dtype=np.float64)
    return types, amounts

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        total_tx = int(input("Number of transactions: "))
        seed_val = int(input("Random seed: "))
    except ValueError:
        return

    random.seed(seed_val)
    ledger = [
        (random.choice(["DEPOSIT", "WITHDRAW"]), round(random.uniform(1.0, 100.0), 2))
        for _ in range(total_tx)
    ]
    types, amounts = build_arrays(ledger)
    workers = os.cpu_count() or 4

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:

        # JIT warmup on tiny input — does not affect the timed run
        print("Warming up Numba JIT...", end=" ", flush=True)
        t = time.perf_counter()
        run_serial_numba(types[:1000], amounts[:1000])
        run_parallel(types[:1000], amounts[:1000], workers, pool)
        print(f"done ({time.perf_counter() - t:.2f}s)")

        # ── Single timed serial run ───────────────────────────────────────────
        t0 = time.perf_counter()
        s_result = round(run_serial_numba(types, amounts), 2)
        serial_time = time.perf_counter() - t0

        # ── Single timed parallel run ─────────────────────────────────────────
        t0 = time.perf_counter()
        p_result = round(run_parallel(types, amounts, workers, pool), 2)
        parallel_time = time.perf_counter() - t0

    print(f"\n{'='*44}")
    print(f"  Transactions : {total_tx:,}  |  Workers: {workers}")
    print(f"{'='*44}")
    print(f"  Serial  : ${s_result:>12.2f}  |  {serial_time*1000:.1f} ms")
    print(f"  Parallel: ${p_result:>12.2f}  |  {parallel_time*1000:.1f} ms")
    print(f"{'='*44}")
    if s_result == p_result:
        print(f"  MATCH ✓  —  {serial_time/parallel_time:.2f}x speedup")
    else:
        print(f"  MISMATCH ✗  serial={s_result}  parallel={p_result}")

if __name__ == "__main__":
    main()