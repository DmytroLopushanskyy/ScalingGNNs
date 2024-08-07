import matplotlib.pyplot as plt
import re
import pandas as pd

# Read the log file
log_file_path = '/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/logs/19-07-run_neo4j_output.log'
with open(log_file_path, 'r') as file:
    log_data = file.readlines()

# Initialize lists to hold the data
losses = []
batch_times = []
batch_numbers = []

# Parse the log data
for line in log_data:
    if 'Batch' in line:
        batch_info = re.search(r'Batch (\d+): (.*)', line)
        if batch_info:
            batch_number = int(batch_info.group(1))
            batch_time = batch_info.group(2)
            batch_numbers.append(batch_number)
            batch_times.append(batch_time)
    if 'Loss' in line:
        loss_info = re.search(r'Loss ([\d\.]+)', line)
        if loss_info:
            losses.append(float(loss_info.group(1)))

# Extract batch times
batch_times = pd.to_datetime(batch_times, format="%Y-%m-%d %H:%M:%S")
batch_durations = batch_times.diff().dropna().astype('timedelta64[s]').tolist()

# Adjust the data lengths to match
batch_numbers = batch_numbers[1:]  # remove the first batch number since diff() reduces length
batch_durations = batch_durations[:len(losses)]

# Convert batch durations to seconds
batch_durations_in_seconds = [duration.total_seconds() for duration in batch_durations]

# Plot Loss for all batches
plt.figure(figsize=(12, 6))
plt.plot(range(len(losses)), losses, marker='o')
plt.title('Loss for all batches')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_plot.png')

# Plot vertical bar chart of time taken for each batch
plt.figure(figsize=(12, 6))
plt.bar(range(len(batch_durations_in_seconds)), batch_durations_in_seconds)
plt.title('Time taken for each batch')
plt.xlabel('Batch Number')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.savefig('time_plot.png')
