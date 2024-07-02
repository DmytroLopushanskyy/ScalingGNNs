#!/bin/bash

# Import data into Neo4j
neo4j-admin database import full papers100m \
    --nodes=PAPER=data/papers100M/nodes_split/nodes_header.csv,data/papers100M/nodes_split/nodes_chunk_.* \
    --relationships=CITES=data/papers100M/edge_idx_split/_edge_header.csv,data/papers100M/edge_idx_split/split.* \
    --normalize-types=false \
    --overwrite-destination=true

# Start Neo4j
neo4j start

# Verify the import
cypher-shell -u neo4j -p password --database=neo4j "MATCH (n) RETURN count(n) as nodes"
cypher-shell -u neo4j -p password --database=neo4j "MATCH ()-[r]->() RETURN count(r) as relationships"

# Stop Neo4j
neo4j stop

