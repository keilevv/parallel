import sys
import time
import threading
import numpy as np
from PIL import Image

# Allow processing very large images (disable decompression bomb check)
Image.MAX_IMAGE_PIXELS = None

def grayscale_sequential(image_np):
    """
    Applies grayscale conversion sequentially using numpy vectorization.
    Grayscale formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
    """
    # Vectorized operation over the whole image
    gray_img = (0.2989 * image_np[:, :, 0] + 
                0.5870 * image_np[:, :, 1] + 
                0.1140 * image_np[:, :, 2]).astype(np.uint8)
    return gray_img

def grayscale_segment(image_np, out_np, start_row, end_row):
    """
    Processes a segment of the image rows for parallel execution using numpy.
    """
    # Each thread processes its slice using vectorized operations
    out_np[start_row:end_row, :] = (0.2989 * image_np[start_row:end_row, :, 0] + 
                                    0.5870 * image_np[start_row:end_row, :, 1] + 
                                    0.1140 * image_np[start_row:end_row, :, 2]).astype(np.uint8)

def grayscale_parallel(image_np, num_threads=4):
    """
    Applies grayscale conversion in parallel using multiple threads.
    Threads share the output numpy array.
    """
    height, width, _ = image_np.shape
    out_np = np.zeros((height, width), dtype=np.uint8)
    
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

    # Sequential Processing
    print("Starting sequential processing...")
    start_time = time.time()
    seq_result = grayscale_sequential(img_np)
    seq_duration = time.time() - start_time
    print(f"Sequential processing finished in: {seq_duration:.4f} seconds")

    # Parallel Processing Analysis
    thread_counts = [1, 2, 4, 8, 16]
    print(f"\n{'Threads':<10} | {'Time (s)':<10} | {'Speedup':<10}")
    print("-" * 35)
    
    # Store results for analysis
    results = []
    
    for tc in thread_counts:
        start_time = time.time()
        par_result = grayscale_parallel(img_np, num_threads=tc)
        par_duration = time.time() - start_time
        speedup = seq_duration / par_duration if par_duration > 0 else 0
        results.append((tc, par_duration, speedup))
        print(f"{tc:<10} | {par_duration:<10.4f} | {speedup:<10.2f}")

    print("-" * 35)
    
    # Save outcomes from the last run (16 threads)
    par_img = Image.fromarray(par_result)
    par_img.save("outcome_parallel.jpg")
    
    seq_img = Image.fromarray(seq_result)
    seq_img.save("outcome_sequential.jpg")
    
    print("\nResults saved as 'outcome_sequential.jpg' and 'outcome_parallel.jpg'")

if __name__ == "__main__":
    main()
