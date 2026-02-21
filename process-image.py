import sys
import time
import threading
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --- SHARED MEMORY RESOURCES ---
# We will share a counter between threads to track progress
progress_lock = threading.Lock()
total_rows_processed = 0
total_rows_to_process = 0

def grayscale_sequential(image_np):
    """
    Optimized sequential processing using np.dot to prevent memory spikes.
    """
    # np.dot is massively more memory-efficient and faster than basic addition
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    gray_img = np.dot(image_np[..., :3], weights).astype(np.uint8)
    return gray_img

def grayscale_segment(image_np, out_np, start_row, end_row):
    """
    Processes a segment of the image and safely shares progress data.
    """
    global total_rows_processed, total_rows_to_process

    # 1. OPTIMIZATION: Use np.dot for the math to prevent OOM kills
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    out_np[start_row:end_row, :] = np.dot(image_np[start_row:end_row, :, :3], weights).astype(np.uint8)

    # 2. SYNCHRONIZATION: Safely update the shared progress tracker
    # Without this lock, threads would overwrite each other's progress updates (Race Condition)
    with progress_lock:
        total_rows_processed += (end_row - start_row)
        progress_percentage = (total_rows_processed / total_rows_to_process) * 100
        # Print safely via the lock so console outputs don't jumble together
        print(f"Thread finished rows {start_row}-{end_row} | Progress: {progress_percentage:.1f}%")

def grayscale_parallel(image_np, num_threads=4):
    global total_rows_processed, total_rows_to_process

    height, width, _ = image_np.shape
    out_np = np.zeros((height, width), dtype=np.uint8)

    # Reset shared variables for this run
    total_rows_processed = 0
    total_rows_to_process = height

    threads = []
    rows_per_thread = height // num_threads

    for i in range(num_threads):
        start_row = i * rows_per_thread
        end_row = height if i == num_threads - 1 else (i + 1) * rows_per_thread

        thread = threading.Thread(target=grayscale_segment, args=(image_np, out_np, start_row, end_row))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return out_np

def main():
    if len(sys.argv) < 2:
        print("Usage: python process-image.py <image_path>")
        return

    image_path = sys.argv[1]

    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print(f"Image loaded: {image_path} ({img_np.shape[1]}x{img_np.shape[0]})")
    print("-" * 30)

    print("Starting sequential processing...")
    start_time = time.time()
    seq_result = grayscale_sequential(img_np)
    seq_duration = time.time() - start_time
    print(f"Sequential processing finished in: {seq_duration:.4f} seconds\n")

    thread_counts = [2, 4, 8, 16]
    print(f"\n{'Threads':<10} | {'Time (s)':<10} | {'Speedup':<10}")
    print("-" * 35)

    for tc in thread_counts:
        print(f"\n--- Running with {tc} threads ---")
        start_time = time.time()
        par_result = grayscale_parallel(img_np, num_threads=tc)
        par_duration = time.time() - start_time
        speedup = seq_duration / par_duration if par_duration > 0 else 0

        print("-" * 35)
        print(f"RESULT: {tc:<2} threads | {par_duration:<10.4f}s | {speedup:<10.2f}x speedup")

if __name__ == "__main__":
    main()
