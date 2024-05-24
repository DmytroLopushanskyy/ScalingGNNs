import os
import kuzu
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from torch_geometric.datasets import Planetoid

# Define paths for saving the combined CSV file
base_path = '../data'
kuzu_path = base_path + '/kuzu'
source_path = kuzu_path + '/source'
combined_path = os.path.join(source_path, 'node_data.csv')
edge_index_path = os.path.join(source_path, 'edge_index.csv')

# Ensure the base directory exists
if not os.path.exists(source_path):
    os.makedirs(source_path)

# Load dataset
cora = Planetoid(root=base_path, name='Cora')[0]

# Prepare the combined data with ID, features, and labels
ids = np.arange(cora.x.shape[0])
features = ['[' + ', '.join(map(str, feat)) + ']' for feat in cora.x.numpy()]
labels = cora.y.numpy()

combined_data = pd.DataFrame({
    'ID': ids,
    'Features': features,
    'Label': labels
})
combined_data.to_csv(combined_path, index=False)

# Convert edge_index to CSV format and save
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
    f'COPY paper FROM "{combined_path}" (HEADER=true)'
)

# Load edges into the Kùzu database
conn.execute(
    f'COPY cites FROM "{edge_index_path}" (HEADER=true)'
)

print("All done! Cora dataset is now loaded into Kùzu.")
