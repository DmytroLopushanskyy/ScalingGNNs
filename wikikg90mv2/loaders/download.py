import psutil
import time
import threading
from ogb.lsc import WikiKG90Mv2Dataset

# Function to load the dataset
def load_dataset():
    dataset = WikiKG90Mv2Dataset(root="./")
    return dataset

# Function to monitor CPU and memory usage
def monitor_resources(interval=5):
    process = psutil.Process()
    while True:
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=None)
        print(f"Memory Usage: {mem_info.rss / (1024 ** 2)} MB")
        print(f"CPU Usage: {cpu_percent}%")
        time.sleep(interval)

# Main function to run the monitoring and dataset loading
def main():
    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True  # Ensure the thread exits when the main program does
    monitor_thread.start()

    # Load the dataset
    load_dataset()

if __name__ == "__main__":
    main()
