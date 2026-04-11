import math, os, random, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit

@njit(cache=True, nogil=True)
def _scan_chunk(types, amounts, start, end, opening_balance):
    bal = opening_balance
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

def build_arrays(ledger):
    types   = np.array([1 if t == "DEPOSIT" else 0 for t, _ in ledger], dtype=np.int8)
    amounts = np.array([a for _, a in ledger], dtype=np.float64)
    return types, amounts

def warmup(types, amounts, workers):
    print("Warming up Numba JIT...", end=" ", flush=True)
    t = time.perf_counter()
    run_serial_numba(types[:1000], amounts[:1000])
    # pre-warm the pool by keeping it alive
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda i: _scan_chunk(types[:1000], amounts[:1000], 0, 1000, 0.0), range(workers)))
    print(f"done ({time.perf_counter() - t:.2f}s)")

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
    warmup(types, amounts, workers)

    n = len(types)
    chunk = math.ceil(n / workers)
    starts = [i * chunk         for i in range(workers)]
    ends   = [min(s + chunk, n) for s in starts]

    # ── Serial baseline ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    s_result = round(run_serial_numba(types, amounts), 2)
    serial_time = time.perf_counter() - t0

    # ── Diagnose each phase individually ─────────────────────────────────────
    with ThreadPoolExecutor(max_workers=workers) as pool:

        # Phase: pool already alive — just measure scan cost
        t0 = time.perf_counter()
        pass1 = list(pool.map(
            lambda i: _scan_chunk(types, amounts, starts[i], ends[i], 0.0),
            range(workers)
        ))
        t_pass1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        openings = [0.0] * workers
        for i in range(1, workers):
            openings[i] = _scan_chunk(types, amounts, starts[i-1], ends[i-1], openings[i-1])
        t_pass2 = time.perf_counter() - t0

        t0 = time.perf_counter()
        finals = list(pool.map(
            lambda i: _scan_chunk(types, amounts, starts[i], ends[i], openings[i]),
            range(workers)
        ))
        t_pass3 = time.perf_counter() - t0

    p_result = round(finals[-1], 2)
    parallel_time = t_pass1 + t_pass2 + t_pass3

    print(f"\n{'='*44}")
    print(f"  Transactions : {total_tx:,}  |  Workers: {workers}")
    print(f"{'='*44}")
    print(f"  Serial            : {serial_time*1000:7.1f} ms")
    print(f"  Parallel total    : {parallel_time*1000:7.1f} ms")
    print(f"    Pass 1 (par)    : {t_pass1*1000:7.1f} ms")
    print(f"    Pass 2 (serial) : {t_pass2*1000:7.1f} ms")
    print(f"    Pass 3 (par)    : {t_pass3*1000:7.1f} ms")
    print(f"{'='*44}")
    if s_result == p_result:
        print(f"  MATCH ✓  —  {serial_time/parallel_time:.2f}x")
    else:
        print(f"  MISMATCH ✗  serial={s_result}  parallel={p_result}")

if __name__ == "__main__":
    main()