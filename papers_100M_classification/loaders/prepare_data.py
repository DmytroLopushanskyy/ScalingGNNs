from multiprocessing import cpu_count
from os import path
from zipfile import ZipFile

import kuzu
import numpy as np
from tqdm import tqdm

unzipped_path = "../data/papers100M/unzipped/"
processed_path = "../data/papers100M/processed/"

# with ZipFile("papers100M-bin.zip", 'r') as papers100M_zip:
#     print('Extracting papers100M-bin.zip...')
#     papers100M_zip.extractall()

# with ZipFile("../data/papers100M/raw/data.npz", 'r') as data_zip:
#     print('Extracting data.npz...')
#     data_zip.extractall()
#
# with ZipFile("../data/papers100M/raw/node-label.npz", 'r') as node_label_zip:
#     print('Extracting node-label.npz...')
#     node_label_zip.extractall()

# print("Converting edge_index to CSV...")
# edge_index = np.load(unzipped_path + 'edge_index.npy', mmap_mode='r')
# csvfile = open(processed_path + 'edge_index.csv', 'w')
# csvfile.write('src,dst\n')
# for i in tqdm(range(edge_index.shape[1])):
#     csvfile.write(str(edge_index[0, i]) + ',' + str(edge_index[1, i]) + '\n')
# csvfile.close()
#
# print("Generating IDs for nodes...")
# node_year = np.load(unzipped_path + 'node_year.npy', mmap_mode='r')
# length = node_year.shape[0]
# ids = np.arange(length)
# np.save(processed_path + 'ids.npy', ids)

ids_path = processed_path + 'ids.npy'
edge_index_path = processed_path + 'edge_index.csv'
node_label_path = unzipped_path + 'node_label.npy'
node_feature_path = unzipped_path + 'node_feat.npy'
node_year_path = unzipped_path + 'node_year.npy'


print("Creating Kùzu database...")
db = kuzu.Database('../data/kuzu')
conn = kuzu.Connection(db, num_threads=cpu_count())
# print("Creating Kùzu tables...")
# conn.execute(
#     "CREATE NODE TABLE paper(id INT64, x FLOAT[128], year INT64, y FLOAT, "
#     "PRIMARY KEY (id));")

try:
    conn.execute("DROP TABLE cites;")
    print("Dropped existing 'cites' table.")
except Exception as e:
    print("No existing 'cites' table found.")

conn.execute("CREATE REL TABLE cites(FROM paper TO paper, MANY_MANY);")

# print("Copying nodes to Kùzu tables...")
# conn.execute('COPY paper FROM ("%s",  "%s",  "%s", "%s") BY COLUMN;' %
#              (ids_path, node_feature_path, node_year_path, node_label_path))
print("Copying edges to Kùzu tables...")
conn.execute('COPY cites FROM "%s";' % (edge_index_path))
print("All done!")
