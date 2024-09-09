import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# Load dataset
dataset = PygNodePropPredDataset(name='ogbn-products', root='./data/dataset/ogbn-products')
graph = dataset[0]  # pyg graph object

# Save node features, labels, and IDs
node_features = graph.x.numpy()
node_labels = graph.y.numpy()
edges = graph.edge_index.t().numpy()

sorted_edge_index = np.sort(edges, axis=1)
unique_edges = np.unique(sorted_edge_index, axis=0)

# Write node CSV for Neo4j
with open('./data/other/nodes.csv', 'w') as f:
    # f.write('id,label,features\n')
    for idx, (features, label) in enumerate(zip(node_features, node_labels)):
        # Convert features array to a string with semicolon-separated floats
        features_str = ';'.join(map(str, features.astype(float)))
        f.write(f"{idx},{label[0]},{features_str}\n")
        if idx % 100_000 == 0:
            print("idx", idx)

# Write edge CSV for Neo4j
with open('./data/other/edges.csv', 'w') as f:
    # f.write('source,target\n')
    idx = 0
    for source, target in unique_edges:
        f.write(f"{source},{target}\n")
        idx += 1
        if idx % 1_000_000 == 0:
            print("idx", idx)

# Initialize Neo4j connection
uri = "bolt://localhost:7687"
user = "neo4j"
password = ""
driver = GraphDatabase.driver(uri, auth=(user, password))

def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def load_nodes(tx, records):
    query = """
    UNWIND $node_list as node
    CREATE (e:PRODUCT {
        id: node.id,
        label: toInteger(node.label),
        features: [x IN split(node.features, ';') | toFloat(x)]
    })
    """
    tx.run(query, node_list=records)

def load_edges(tx, records):
    query = """
    UNWIND $edge_list as edge
    MATCH (source:PRODUCT {id: edge.source})
    MATCH (target:PRODUCT {id: edge.target})
    MERGE (source)-[r:LINK]-(target)
    """
    tx.run(query, edge_list=records)

def batch_process(dataframe, func):
    batch_len = 10_000
    for batch_start in tqdm(range(0, len(dataframe), batch_len)):
        batch_end = batch_start + batch_len
        records = dataframe.iloc[batch_start:batch_end].to_dict("records")
        with driver.session() as session:
            session.execute_write(func, records)


# Load nodes and edges with a clean start
with driver.session() as session:
    session.execute_write(clear_database)


node_list = pd.read_csv("./data/other/nodes.csv", header=None, names=['id', 'label', 'features'], dtype=str)
edge_list = pd.read_csv("./data/other/edges.csv", header=None, names=['source', 'target'], dtype=str)

batch_process(node_list, load_nodes)
batch_process(edge_list, load_edges)

driver.close()

