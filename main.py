import numpy as np
import time
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass

# =========================
# Configuration
# =========================

DATASET_SIZE = 20000        # number of students
FEATURES = 6               # academic features
EPOCHS = 300               # training iterations
LEARNING_RATE = 0.01
N_RUNS = 8                 # independent training runs
PROCESS_COUNTS = [1, 2, 3, 4] # change based on your CPU

np.random.seed(42)

# =========================
# Data Simulation
# =========================

def generate_dataset(n_samples):
    """
    Simulates student academic data for UTB.
    """
    gpa = np.random.normal(3.2, 0.4, n_samples)
    attendance = np.random.uniform(0.6, 1.0, n_samples)
    failed_courses = np.random.poisson(1.2, n_samples)
    credits = np.random.randint(12, 24, n_samples)
    socioeconomic = np.random.randint(1, 6, n_samples)
    tutoring = np.random.binomial(1, 0.3, n_samples)

    X = np.column_stack([
        gpa,
        attendance,
        failed_courses,
        credits,
        socioeconomic,
        tutoring
    ])

    # Risk probability (hidden ground truth)
    logits = (
        -1.5 * gpa
        -2.0 * attendance
        +1.2 * failed_courses
        -0.02 * credits
        +0.4 * socioeconomic
        -0.6 * tutoring
    )

    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y

# =========================
# Logistic Regression (From Scratch)
# =========================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, epochs, lr):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        linear = np.dot(X, weights) + bias
        preds = sigmoid(linear)

        dw = (1 / n_samples) * np.dot(X.T, (preds - y))
        db = (1 / n_samples) * np.sum(preds - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias

# =========================
# Training Task
# =========================

def training_task(seed):
    np.random.seed(seed)
    X, y = generate_dataset(DATASET_SIZE)

    # simple train/validation split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    train_logistic_regression(
        X_train,
        y_train,
        EPOCHS,
        LEARNING_RATE
    )
    
    # Return class distribution for this run
    return np.bincount(y, minlength=2)

# =========================
# Serial Execution
# =========================

def run_serial():
    start = time.perf_counter()
    all_counts = []
    for i in range(N_RUNS):
        counts = training_task(i)
        all_counts.append(counts)
    end = time.perf_counter()
    total_distribution = np.sum(all_counts, axis=0)
    return end - start, total_distribution

# =========================
# Parallel Execution
# =========================

def run_parallel(n_processes):
    start = time.perf_counter()
    with Pool(processes=n_processes) as pool:
        all_counts = pool.map(training_task, range(N_RUNS))
    end = time.perf_counter()
    total_distribution = np.sum(all_counts, axis=0)
    return end - start, total_distribution

# =========================
# Metrics
# =========================

@dataclass
class Metrics:
    processes: int
    time: float
    speedup: float
    efficiency: float
    overhead: float

# =========================
# Main Experiment
# =========================

def main():
    print("\n=== UTB Academic Risk Prediction ===\n")
    print(f"Dataset size: {DATASET_SIZE}")
    print(f"Training runs: {N_RUNS}")
    print(f"Epochs per run: {EPOCHS}")
    print(f"CPU cores available: {cpu_count()}\n")

    serial_time, distribution = run_serial()
    print(f"Serial execution time: {serial_time:.4f} s\n")

    results = []

    for p in PROCESS_COUNTS:
        parallel_time, _ = run_parallel(p)
        speedup = serial_time / parallel_time
        efficiency = speedup / p
        overhead = parallel_time - (serial_time / p)

        results.append(
            Metrics(
                processes=p,
                time=parallel_time,
                speedup=speedup,
                efficiency=efficiency,
                overhead=overhead
            )
        )

    print("Processes | Time (s) | Speedup | Efficiency | Overhead (s)")
    print("-" * 60)

    for r in results:
        print(
            f"{r.processes:^9} | "
            f"{r.time:^8.4f} | "
            f"{r.speedup:^7.2f} | "
            f"{r.efficiency:^9.2f} | "
            f"{r.overhead:^11.4f}"
        )

    print("\n" + "=" * 45)
    print("      STUDENT CLASS DISTRIBUTION")
    print("=" * 45)
    print(f"Per Run (Dataset Size: {DATASET_SIZE:,})")
    print(f"  - Low Risk:    {int(distribution[0]/N_RUNS):,}")
    print(f"  - Dropout Risk: {int(distribution[1]/N_RUNS):,}")
    print("-" * 45)
    print(f"Experiment Total ({N_RUNS} Runs)")
    print(f"  - Total Low Risk:    {distribution[0]:,}")
    print(f"  - Total Dropout Risk: {distribution[1]:,}")
    print(f"  - Total Processed:    {np.sum(distribution):,}")
    print("=" * 45)

    print("\nExperiment completed.\n")

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()
