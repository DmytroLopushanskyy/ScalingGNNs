import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Paths to the log files for different nodes
log_file_paths = [
    "/data/project/anonym/ScalingGNNs/distributed_training/distributed-neo4j/logs/experiment-256GB-RAM-48CPUs-node-0.log",
    "/data/project/anonym/ScalingGNNs/distributed_training/distributed-neo4j/logs/experiment-256GB-RAM-48CPUs-node-1.log",
    # Add more paths as needed
]

# Define regex patterns
train_pattern = r"\[Node (\d+)\] Train: epoch=(\d+), it=(\d+), loss=([\d.]+), time=([\d.]+)"
test_pattern = r"\[Node (\d+)\] Test: epoch=(\d+), it=(\d+), acc=([\d.]+), time=([\d.]+)"
train_end_pattern = r"\[Node (\d+)\] Train epoch (\d+) END: loss=([\d.]+), time=([\d.]+)"
test_end_pattern = r"\[Node (\d+)\] Test epoch (\d+) END: acc=([\d.]+), time=([\d.]+)"

# Initialize data storage
train_iterations = []
test_iterations = []
train_endings = []
test_endings = []

# Read and parse log data
for file_path in log_file_paths:
    with open(file_path, "r") as file:
        log_data = file.read()
    
    train_iterations.extend(re.findall(train_pattern, log_data))
    test_iterations.extend(re.findall(test_pattern, log_data))
    train_endings.extend(re.findall(train_end_pattern, log_data))
    test_endings.extend(re.findall(test_end_pattern, log_data))

# Adjust iterations and store data for each node
train_data = {}
test_data = {}
train_stats = {}
test_stats = {}
epoch_total_times = {}

for node, epoch, it, loss, time in train_iterations:
    node = int(node)
    if node not in train_data:
        train_data[node] = {"iterations": [], "losses": [], "epoch_starts": {}}
    epoch = int(epoch)
    it = int(it)
    if it == 0:
        train_data[node]["epoch_starts"][epoch] = len(train_data[node]["iterations"])
    train_data[node]["iterations"].append((len(train_data[node]["iterations"]), float(time)))
    train_data[node]["losses"].append((len(train_data[node]["iterations"]), float(loss)))

for node, epoch, it, acc, time in test_iterations:
    node = int(node)
    if node not in test_data:
        test_data[node] = {"iterations": [], "epoch_starts": {}}
    epoch = int(epoch)
    it = int(it)
    if it == 0:
        test_data[node]["epoch_starts"][epoch] = len(test_data[node]["iterations"])
    test_data[node]["iterations"].append((len(test_data[node]["iterations"]), float(time)))

for node, epoch, loss, time in train_endings:
    node = int(node)
    if node not in train_stats:
        train_stats[node] = []
    train_stats[node].append(f'Epoch {epoch}: loss={loss}, time={time}')
    epoch = int(epoch)
    time = float(time)
    if node not in epoch_total_times:
        epoch_total_times[node] = {}
    if epoch not in epoch_total_times[node]:
        epoch_total_times[node][epoch] = 0
    epoch_total_times[node][epoch] += time

for node, epoch, acc, time in test_endings:
    node = int(node)
    if node not in test_stats:
        test_stats[node] = []
    test_stats[node].append(f'Epoch {epoch}: acc={acc}, time={time}')

# Rearrange train and test stats for final display
test_stats_final = {}
train_stats_final = {}
for node in sorted(test_stats.keys()):
    for entry in test_stats[node]:
        epoch = int(re.search(r'Epoch (\d+):', entry).group(1))
        acc = re.search(r'acc=([\d.]+)', entry).group(1)
        if epoch not in test_stats_final:
            test_stats_final[epoch] = {}
        test_stats_final[epoch][node] = acc

for node in sorted(epoch_total_times.keys()):
    for epoch, time in epoch_total_times[node].items():
        if epoch not in train_stats_final:
            train_stats_final[epoch] = {}
        train_stats_final[epoch][node] = time

# Plotting
fig = plt.figure(figsize=(24, 12))
num_nodes = len(log_file_paths)
gs = GridSpec(2, 4)

def add_epoch_xaxis(ax):
    ax_epoch = ax.twiny()
    ax_epoch.set_xticks([])
    ax_epoch.set_xticklabels([])
    ax_epoch.set_xlabel('Epoch')

# Define colors for each node
colors = ['blue', 'green', 'gray', 'black', 'purple', 'red', 'orange', 'yellow', 'pink', 'cyan', 'magenta', 'lime', 'brown', 'teal', 'navy', 'maroon']

# Train iterations plot
ax0 = fig.add_subplot(gs[0, :2])
for node, data in train_data.items():
    iters, times = zip(*data["iterations"])
    ax0.plot(iters, times, label=f'Node {node} Train Time', color=colors[node % len(colors)])
for epoch, epoch_start in train_data[node]["epoch_starts"].items():
    ax0.axvline(x=epoch_start, color='red', linestyle='--', linewidth=0.5)
    ax0.text(epoch_start, max(times), f'{epoch}', rotation=0, verticalalignment='bottom', horizontalalignment='center')
ax0.set_xlabel('Iteration')
ax0.set_ylabel('Time (s)')
ax0.set_title('Train Iteration Time')
ax0.legend()
add_epoch_xaxis(ax0)

# Test iterations plot
ax1 = fig.add_subplot(gs[0, 2:])
for node, data in test_data.items():
    iters, times = zip(*data["iterations"])
    ax1.plot(iters, times, label=f'Node {node} Test Time', color=colors[node % len(colors)])
for epoch, epoch_start in test_data[node]["epoch_starts"].items():
    ax1.axvline(x=epoch_start, color='red', linestyle='--', linewidth=0.5)
    ax1.text(epoch_start, max(times), f'{epoch}', rotation=0, verticalalignment='bottom', horizontalalignment='center')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Time (s)')
ax1.set_title('Test Iteration Time')
ax1.legend()
add_epoch_xaxis(ax1)

# Train loss plot (50% width)
ax2 = fig.add_subplot(gs[1, :2])
for node, data in train_data.items():
    iters, losses = zip(*data["losses"])
    ax2.plot(iters, losses, label=f'Node {node} Train Loss', color=colors[node % len(colors)])
for epoch, epoch_start in train_data[node]["epoch_starts"].items():
    ax2.axvline(x=epoch_start, color='red', linestyle='--', linewidth=0.5)
    ax2.text(epoch_start, max(losses), f'{epoch}', rotation=0, verticalalignment='bottom', horizontalalignment='center')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.set_title('Train Loss Over Iterations')
ax2.legend()
add_epoch_xaxis(ax2)

# Display test statistics (25% width)
ax3 = fig.add_subplot(gs[1, 2])
ax3.axis('off')
stat_text = f"Test Stats (Node {'/Node '.join(map(str, range(num_nodes)))}):\n"
for epoch in sorted(test_stats_final.keys()):
    acc_values = [test_stats_final[epoch].get(node, '-') for node in range(num_nodes)]
    stat_text += f'Epoch {epoch}: acc={" / ".join(acc_values)}\n'
ax3.text(0.5, 0.5, stat_text, ha='center', va='center', fontsize=12)

# Display total train time per epoch for each node (25% width)
ax4 = fig.add_subplot(gs[1, 3])
ax4.axis('off')
train_time_text = f"Train Stats (Node {'/Node '.join(map(str, range(num_nodes)))}):\n"
for epoch in sorted(train_stats_final.keys()):
    time_values = [f'{train_stats_final[epoch].get(node, 0):.2f}s' for node in range(num_nodes)]
    train_time_text += f'Epoch {epoch}: time={" / ".join(time_values)}\n'
ax4.text(0.5, 0.5, train_time_text, ha='center', va='center', fontsize=12)

fig.suptitle('Training and Testing Metrics\n2 nodes (same machine), 256 GB RAM, 48 CPUs, 6 hours runtime\n num_neighbors: [15,10,5], 4 sampler sub-processes, 10 threads for each sampler sub-process, max 4 concurrent RPC for each sampler', fontsize=16)
fig.subplots_adjust(top=0.92)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("train_test_iteration_times_and_statistics_rearranged.png")
plt.show()
