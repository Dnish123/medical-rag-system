"""
Pinecone vector database operations.
Handles connection, indexing, and retrieval.
"""

from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
from utils.config import Config
import time

class PineconeDB:
    """
    Wrapper for Pinecone operations.
    """

    def __init__(self):
        """Initialize Pinecone connection"""
        # Updated initialization for newer Pinecone versions
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self.index = None

        # Create index if doesn't exist
        self._setup_index()

    def _setup_index(self):
        """Create or connect to Pinecone index"""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        except:
            # Fallback for older API
            existing_indexes = self.pc.list_indexes().names()

        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=Config.PINECONE_DIMENSION,
                metric=Config.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            time.sleep(10)

        self.index = self.pc.Index(self.index_name)
        print(f"✅ Connected to index: {self.index_name}")

    def upsert_vectors(self, vectors: List[Dict], batch_size: int = 100):
        """
        Upload vectors to Pinecone in batches.

        Args:
            vectors: List of vector dictionaries
            batch_size: Batch size for upload
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

        print(f"✅ Upserted {len(vectors)} vectors")

    def query(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Query similar vectors from Pinecone.

        Args:
            query_vector: Query embedding
            top_k: Number of results to return

        Returns:
            List of matched documents with metadata
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        # Format results
        docs = []
        for match in results['matches']:
            docs.append({
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'book': match['metadata'].get('book', ''),
                'page': match['metadata'].get('page', 0),
                'paragraph': match['metadata'].get('paragraph', 0)
            })

        return docs

    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        # Handle both dict and object responses
        if hasattr(stats, 'total_vector_count'):
            return {'total_vector_count': stats.total_vector_count}
        return stats

    def delete_all(self):
        """Delete all vectors from index"""
        self.index.delete(delete_all=True)
        print("✅ Deleted all vectors from index")

    def check_if_populated(self) -> bool:
        """Check if index has vectors"""
        stats = self.get_index_stats()
        count = stats.get('total_vector_count', 0)
        return count > 0