import numpy as np
from torch_geometric.datasets import Planetoid
import pandas as pd
from neo4j import GraphDatabase

# Load dataset
cora = Planetoid(root='./data', name='Cora')[0]

# Save node features, labels, and IDs
node_features = cora.x.numpy()
node_labels = cora.y.numpy()
ids = np.arange(cora.x.shape[0])
edges = cora.edge_index.t().numpy()

# Write node CSV for Neo4j
with open('./data/nodes.csv', 'w') as f:
    f.write('id,label,features\n')
    for idx, (features, label) in enumerate(zip(node_features, node_labels)):
        # Convert features array to a string
        features_str = ';'.join(map(str, features))
        f.write(f"{idx},{label},{features_str}\n")

# Write edge CSV for Neo4j
with open('./data/edges.csv', 'w') as f:
    f.write('source,target\n')
    for source, target in edges:
        f.write(f"{source},{target}\n")

# Initialize Neo4j connection
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(user, password))

def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def load_nodes(tx, records):
    query = """
    UNWIND $node_list as node
    CREATE (e:Paper {
        ID: toInteger(node.id),
        label: toInteger(node.label),
        features: node.features
    })
    """
    tx.run(query, node_list=records)

def load_edges(tx, records):
    query = """
    UNWIND $edge_list as edge
    MATCH (source:Paper {ID: toInteger(edge.source)})
    MATCH (target:Paper {ID: toInteger(edge.target)})
    MERGE (source)-[r:CITES]->(target)
    """
    tx.run(query, edge_list=records)

def batch_process(dataframe, func):
    batch_len = 500
    for batch_start in range(0, len(dataframe), batch_len):
        batch_end = batch_start + batch_len
        records = dataframe.iloc[batch_start:batch_end].to_dict("records")
        with driver.session() as session:
            session.execute_write(func, records)

# Load nodes and edges with a clean start
with driver.session() as session:
    session.execute_write(clear_database)

node_list = pd.read_csv("./data/nodes.csv")
edge_list = pd.read_csv("./data/edges.csv")

batch_process(node_list, load_nodes)
batch_process(edge_list, load_edges)

driver.close()
