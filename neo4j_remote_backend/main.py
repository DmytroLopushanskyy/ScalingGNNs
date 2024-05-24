from __future__ import annotations

from feature_store import Neo4jFeatureStore
from graph_store import Neo4jGraphStore

class Neo4jClient:
    def __init__(self):
        pass

    def get_torch_geometric_remote_backend(
            self, num_threads: int | None = None
    ) -> tuple[Neo4jFeatureStore, Neo4jGraphStore]:
        return (
            Neo4jFeatureStore(num_threads),
            Neo4jGraphStore(num_threads),
        )
