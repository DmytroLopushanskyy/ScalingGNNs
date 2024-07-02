from multiprocessing import cpu_count
from os import path
from zipfile import ZipFile
from os import path, listdir
import kuzu
import gc
import numpy as np
from tqdm import tqdm
from datetime import datetime

unzipped_path = "data/papers100M/unzipped/"
processed_path = "data/papers100M/processed/"
edge_idx_split_path = "data/papers100M/edge_idx_split/"

with ZipFile("papers100M-bin.zip", 'r') as papers100M_zip:
    print('Extracting papers100M-bin.zip...')
    papers100M_zip.extractall()

with ZipFile("../data/papers100M/raw/data.npz", 'r') as data_zip:
    print('Extracting data.npz...')
    data_zip.extractall()

with ZipFile("../data/papers100M/raw/node-label.npz", 'r') as node_label_zip:
    print('Extracting node-label.npz...')
    node_label_zip.extractall()

print("Converting edge_index to CSV...")
edge_index = np.load(unzipped_path + 'edge_index.npy', mmap_mode='r')
csvfile = open(processed_path + 'edge_index.csv', 'w')
csvfile.write('src,dst\n')
for i in tqdm(range(edge_index.shape[1])):
    csvfile.write(str(edge_index[0, i]) + ',' + str(edge_index[1, i]) + '\n')
csvfile.close()

print("Generating IDs for nodes...")
node_year = np.load(unzipped_path + 'node_year.npy', mmap_mode='r')
length = node_year.shape[0]
ids = np.arange(length)
np.save(processed_path + 'ids.npy', ids)

ids_path = processed_path + 'ids.npy'                # 848M
edge_index_path = processed_path + 'edge_index.csv'  # 28G
node_label_path = unzipped_path + 'node_label.npy'   # 424M
node_feature_path = unzipped_path + 'node_feat.npy'  # 53G
node_year_path = unzipped_path + 'node_year.npy'     # 848M

print("Creating Kùzu database...")
start_time = datetime.now()
db = kuzu.Database('data/kuzu-4')
conn = kuzu.Connection(db, num_threads=cpu_count())
print(f"Database creation completed in {datetime.now() - start_time}")

print("Creating Kùzu tables...")
start_time = datetime.now()
conn.execute(
    "CREATE NODE TABLE paper(id INT64, x FLOAT[128], year INT64, y FLOAT, "
    "PRIMARY KEY (id));")
print(f"Table creation completed in {datetime.now() - start_time}")

try:
    start_time = datetime.now()
    conn.execute("DROP TABLE cites;")
    print(f"Dropped existing 'cites' table in {datetime.now() - start_time}")
except Exception as e:
    print(f"No existing 'cites' table found: {e}")

start_time = datetime.now()
conn.execute("CREATE REL TABLE cites(FROM paper TO paper, MANY_MANY);")
print(f"Relationship table creation completed in {datetime.now() - start_time}")

print("Copying nodes to Kùzu tables...")
start_time = datetime.now()
conn.execute('COPY paper FROM ("%s",  "%s",  "%s", "%s") BY COLUMN;' %
             (ids_path, node_feature_path, node_year_path, node_label_path))
print(f"Nodes copied in {datetime.now() - start_time}")

conn.close()
conn = kuzu.Connection(db, num_threads=cpu_count())

# Read and copy edges from split CSV files
split_files = [f for f in listdir(edge_idx_split_path) if f.startswith('split_')]

print("Copying edges to Kùzu tables from split files...")
for split_file in tqdm(split_files, desc="Processing split files"):
    start_time = datetime.now()
    file_path = path.join(edge_idx_split_path, split_file)
    print("file_path", file_path, "start_time", start_time)
    conn.execute('COPY cites FROM "%s";' % file_path)
    print(f"Copied {file_path} in {datetime.now() - start_time}")
    
    conn.close()
    db.close()
    del db
    del conn
    gc.collect()

    db = kuzu.Database('data/kuzu-4')
    conn = kuzu.Connection(db, num_threads=cpu_count())

print("All done!")
