import kuzu

# Define the path to your Kùzu database
kuzu_path = '../data/kuzu'

# Create or connect to a Kùzu database
db = kuzu.Database(kuzu_path)
conn = kuzu.Connection(db)

# Cypher-like query to fetch all node features and labels
node_query = """
MATCH (p:paper)
RETURN p.id AS id, p.x AS x, p.y AS y
"""

# Cypher-like query to fetch first 10 edges
edge_query = """
MATCH (p1:paper)-[:cites]->(p2:paper)
RETURN p1.id AS src, p2.id AS dst
LIMIT 10
"""

# Execute the queries
node_result = conn.execute(node_query)
edge_result = conn.execute(edge_query)

# Print all node features and their corresponding labels
print("Node Features and Labels:")
ones_count = 0
nodes_count = 0
while node_result.has_next():
    row = node_result.get_next()
    print(f"Node ID: {row[0]}")
    print("Features (Non-zero indices):")
    features = row[1]  # Assuming the second element is the feature array
    non_zero_indices = [i for i, value in enumerate(features) if value != 0]
    print(non_zero_indices)
    ones_count += len(non_zero_indices)
    nodes_count += 1
    print(f"Label: {row[2]}")  # Assuming the third element is the label
    print("-----")
print(ones_count)
print(nodes_count)
# Print first 10 edges to verify connections
print("Sample Edges:")
while edge_result.has_next():
    row = edge_result.get_next()
    print(f"Source: {row[0]}, Destination: {row[1]}")  # Assuming the format is source, destination

# Close the connection
conn.close()
print("Query execution completed.")