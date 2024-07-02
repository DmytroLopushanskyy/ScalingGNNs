import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Paths to your files
UNZIPPED_PATH = "data/papers100M/unzipped/"
PROCESSED_PATH = "data/papers100M/processed/"
DESTINATION_PATH = "data/papers100M/nodes_split/"
CHUNK_SIZE = 1_000_000  # 1 mln

# Create necessary directories
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def chunk_array(np_array, chunk_size):
    for i in range(0, np_array.shape[0], chunk_size):
        yield np_array[i:i + chunk_size]

def convert_npy_to_csv():
    logging.info("Converting npy files to CSV...")

    logging.info(f"Reading files now.")

    # Load npy files
    node_year = np.load(os.path.join(UNZIPPED_PATH, 'node_year.npy'))       # 848M
    logging.info(f"Loaded node_year")
    node_label = np.load(os.path.join(UNZIPPED_PATH, 'node_label.npy'))     # 424M
    logging.info(f"Loaded node_label")
    ids = np.load(os.path.join(PROCESSED_PATH, 'ids.npy'))                  # 848M
    logging.info(f"Loaded ids")
    node_feat = np.load(os.path.join(UNZIPPED_PATH, 'node_feat.npy'))       # 53G
    logging.info(f"Loaded node_feat")

    nan_count = np.isnan(node_label).sum()
    non_nan_count = np.count_nonzero(~np.isnan(node_label))
    logging.info(f"Found {nan_count} NaN values and {non_nan_count} non-NaN values in node_label")

    total_nodes = node_year.shape[0]
    chunks = chunk_array(np.arange(total_nodes), CHUNK_SIZE)

    for i, chunk in enumerate(chunks):
        features_str = [';'.join(map(str, node_feat[idx])) for idx in chunk]
        year_flat = node_year[chunk].flatten()
        label_flat = node_label[chunk].flatten()
        
        df = pd.DataFrame({
            'id': ids[chunk],
            'year': year_flat,
            'label': label_flat,
            'features': features_str
        })

        chunk_file = os.path.join(DESTINATION_PATH, f'nodes_chunk_{i}.csv')
        df.to_csv(chunk_file, index=False, header=False)
        logging.info(f"Processed {chunk_file}")

convert_npy_to_csv()
