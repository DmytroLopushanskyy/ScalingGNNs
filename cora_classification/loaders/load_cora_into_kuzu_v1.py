import os
import kuzu
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from torch_geometric.datasets import Planetoid

# Define paths for saving numpy files
base_path = './cora_classification/data'
kuzu_path = base_path + '/kuzu'
source_path = kuzu_path + '/source'
node_features_path = os.path.join(source_path, 'node_features.npy')
node_labels_path = os.path.join(source_path, 'node_labels.npy')
edge_index_path = os.path.join(source_path, 'edge_index.csv')
ids_path = os.path.join(source_path, 'ids.npy')

# Ensure the base directory exists
if not os.path.exists(source_path):
    os.makedirs(source_path)
#
# node_features = np.load(node_features_path)
# print(len(node_features))
# cc = 0
# for row in node_features:
#     # print(row)
#     lst = (row != 0).nonzero()[0].tolist()
#     cc += len(lst)
#     print(lst)
# print(cc)
# exit(1)
# node_labels = np.load(node_labels_path)[:10]
# ids = np.load(ids_path)[:100]
# edge_index = pd.read_csv(edge_index_path).head(10)

# print(x_preview)
# print(y_preview, ids_preview)
# print(edges_preview)

# Load dataset
cora = Planetoid(root=base_path, name='Cora')[0]

# Save node features and labels
np.save(node_features_path, cora.x.numpy())
np.save(node_labels_path, cora.y.numpy())

# Generate and save IDs for nodes
ids = np.arange(cora.x.shape[0])
np.save(ids_path, ids)

# Convert edge_index to CSV format for Kùzu
edges = cora.edge_index.numpy()
edges_df = pd.DataFrame(edges.T, columns=['src', 'dst'])
edges_df.to_csv(edge_index_path, index=False)

# Create or connect to a Kùzu database
db = kuzu.Database(kuzu_path)
conn = kuzu.Connection(db, num_threads=cpu_count())

# Drop tables if they exist
try:
    conn.execute("DROP TABLE cites")
    conn.execute("DROP TABLE paper")
except:
    pass

# Create tables in Kùzu database
conn.execute(
    "CREATE NODE TABLE paper"
    "(id INT64, x FLOAT[1433], y INT64, "
    "PRIMARY KEY (id))"
)
conn.execute(
    "CREATE REL TABLE cites"
    "(FROM paper TO paper, MANY_MANY)"
)

# Load nodes into the Kùzu database
conn.execute(
    f'COPY paper FROM ("{ids_path}", "{node_features_path}", "{node_labels_path}") BY COLUMN'
)

# Load edges into the Kùzu database
conn.execute(
    f'COPY cites FROM "{edge_index_path}" (HEADER=true)'
)

print("All done! Cora dataset is now loaded into Kùzu.")