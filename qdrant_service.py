import os
from typing import List, Dict, Any, Optional, Iterator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Record
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, SnapshotDescription
)

# Load environment variables
load_dotenv()


class CreateCollectionParams(BaseModel):
    """Parameters required for creating a Qdrant collection"""
    collection_name: str = Field(..., description="Name of the collection to create")
    vector_size: int = Field(..., description="Size of the vectors")
    distance_metric: str = Field("COSINE", description="Distance metric (COSINE, EUCLIDEAN, DOT)")


class UpsertVectorsParams(BaseModel):
    """Parameters for inserting or updating vectors in Qdrant"""
    collection_name: str = Field(..., description="Name of the collection")
    vectors: List[List[float]] = Field(..., description="List of vectors to insert/update")
    payloads: List[Dict[str, Any]] = Field(..., description="Metadata associated with each vector")


class SearchVectorsParams(BaseModel):
    """Parameters for searching vectors in Qdrant"""
    collection_name: str = Field(..., description="Name of the collection")
    query_vector: List[float] = Field(..., description="Vector to search for")
    limit: int = Field(20, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for the search")


class RetrievePointsParams(BaseModel):
    """Parameters for retrieving specific points"""
    collection_name: str = Field(..., description="Name of the collection")
    ids: List[int] = Field(..., description="List of point IDs to retrieve")


class DeletePointsParams(BaseModel):
    """Parameters for deleting points from a collection"""
    collection_name: str = Field(..., description="Name of the collection")
    ids: List[int] = Field(..., description="List of point IDs to delete")


class QdrantService:
    """Handles interactions with Qdrant for vector storage and retrieval"""

    def __init__(self, api_key: str = None, host: str = None):
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.host = host or os.getenv("QDRANT_HOST")

        if not self.api_key:
            raise ValueError("Qdrant API key must be provided.")
        if not self.host:
            raise ValueError("Qdrant host URL must be provided.")

        self.client = QdrantClient(url=self.host, api_key=self.api_key)

    def create_collection(self, params: CreateCollectionParams):
        """Creates a new Qdrant collection"""
        if not self.client.collection_exists(params.collection_name):
            self.client.create_collection(
                collection_name=params.collection_name,
                vectors_config=VectorParams(
                    size=params.vector_size,
                    distance=Distance[params.distance_metric]
                )
            )

    def delete_collection(self, collection_name: str):
        """Deletes a Qdrant collection"""
        self.client.delete_collection(collection_name)

    def list_collections(self) -> List[str]:
        """Lists all available Qdrant collections"""
        return [collection.name for collection in self.client.list_collections().collections]

    def upsert_vectors(self, params: UpsertVectorsParams):
        """Inserts or updates vectors in a Qdrant collection"""
        points = [
            PointStruct(id=idx, vector=vector, payload=payload)
            for idx, (vector, payload) in enumerate(zip(params.vectors, params.payloads))
        ]
        self.client.upsert(collection_name=params.collection_name, points=points)

    def search_vectors(self, params: SearchVectorsParams):
        """Searches for similar vectors in a Qdrant collection"""
        query_filter = None
        if params.filters:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key=key,
                        range=Range(**value)
                    ) for key, value in params.filters.items()
                ]
            )
        print(params)
        return self.client.query_points(
            collection_name=params.collection_name,
            query=params.query_vector,
            query_filter=query_filter,
            limit=params.limit
        )

    def retrieve_points(self, params: RetrievePointsParams) -> list[Record]:
        """Retrieves specific points by ID"""
        return self.client.retrieve(collection_name=params.collection_name, ids=params.ids)

    def delete_points(self, params: DeletePointsParams):
        """Deletes points by ID"""
        self.client.delete(collection_name=params.collection_name, ids=params.ids)

    def count_points(self, collection_name: str) -> int:
        """Counts the number of points in a collection"""
        return self.client.count(collection_name=collection_name).count

    def scroll_points(self, collection_name: str) -> Iterator[PointStruct]:
        """Scrolls through all points in a collection"""
        yield from self.client.scroll(collection_name=collection_name)

    def create_snapshot(self, collection_name: str) -> SnapshotDescription:
        """Creates a snapshot of a Qdrant collection"""
        return self.client.create_snapshot(collection_name=collection_name)

    def list_snapshots(self, collection_name: str) -> List[str]:
        """Lists available snapshots for a collection"""
        return [snapshot.name for snapshot in self.client.list_snapshots(collection_name=collection_name).snapshots]

    def delete_snapshot(self, collection_name: str, snapshot_name: str):
        """Deletes a snapshot of a collection"""
        self.client.delete_snapshot(collection_name=collection_name, snapshot_name=snapshot_name)

    def cluster_info(self) -> Dict[str, Any]:
        """Retrieves Qdrant cluster information"""
        return self.client.cluster_info()

    def update_cluster(self, new_config: Dict[str, Any]):
        """Updates Qdrant cluster configuration"""
        self.client.update_cluster(new_config)


# Example Usage
if __name__ == "__main__":
    qdrant_service = QdrantService()

    # Create a collection
    qdrant_service.create_collection(CreateCollectionParams(
        collection_name="documents",
        vector_size=768,
        distance_metric="COSINE"
    ))

    # Upsert vectors
    import numpy as np
    vectors = np.random.rand(10, 768).tolist()
    payloads = [{"metadata": f"vector_{i}"} for i in range(10)]
    qdrant_service.upsert_vectors(UpsertVectorsParams(
        collection_name="documents",
        vectors=vectors,
        payloads=payloads
    ))

    # Search for similar vectors
    query_vector = np.random.rand(768).tolist()
    results = qdrant_service.search_vectors(SearchVectorsParams(
        collection_name="documents",
        query_vector=query_vector
    ))
    print("Search Results:", results)

    # Retrieve points
    retrieved_points = qdrant_service.retrieve_points(RetrievePointsParams(
        collection_name="documents",
        ids=[0, 1, 2]
    ))
    print("Retrieved Points:", retrieved_points)

    # Count points in collection
    print("Total Points in Collection:", qdrant_service.count_points("documents"))

    # Delete a collection
    qdrant_service.delete_collection("documents")
