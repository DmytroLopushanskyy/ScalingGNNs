import torch
import time
from ogb.nodeproppred import PygNodePropPredDataset

def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def extract_split_indices():
    # Load the dataset
    print(f"[{current_time()}] Start loading")
    start_time = time.time()
    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    load_time = time.time()

    print(f"[{current_time()}] Dataset loaded in {load_time - start_time:.2f} seconds")

    split_idx = dataset.get_idx_split()
    extract_time = time.time()
    
    print(f"[{current_time()}] Split indices extracted in {extract_time - load_time:.2f} seconds")
    
    # Extract train, validation, and test indices
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    print(f"[{current_time()}] Number of training nodes: {len(train_idx)}")
    print(f"[{current_time()}] Number of validation nodes: {len(valid_idx)}")
    print(f"[{current_time()}] Number of test nodes: {len(test_idx)}")
    
    # Total time taken
    total_time = time.time()
    print(f"[{current_time()}] Total time taken: {total_time - start_time:.2f} seconds")
    
    return train_idx, valid_idx, test_idx

if __name__ == "__main__":
    train_idx, valid_idx, test_idx = extract_split_indices()
