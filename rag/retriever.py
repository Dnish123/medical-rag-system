"""
Retrieval pipeline for fetching relevant document chunks.
"""

from vectorstore.pinecone_db import PineconeDB
from embeddings.embedder import ChunkEmbedder
from typing import List, Dict, Optional
from utils.config import Config


class RAGRetriever:
    """
    Handles document retrieval from Pinecone.
    """

    def __init__(self):
        """Initialize retriever components"""
        self.embedder = ChunkEmbedder()
        self.vectorstore = PineconeDB()

        # Verify index is populated
        if not self.vectorstore.check_if_populated():
            raise ValueError(
                "Pinecone index is empty. Please run admin/ingest_books.py first."
            )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User question
            top_k: Number of documents to retrieve (default: Config.TOP_K)

        Returns:
            List of relevant document chunks
        """
        if top_k is None:
            top_k = Config.TOP_K

        # Generate query embedding
        query_vector = self.embedder.embed_query(query)

        # Retrieve from Pinecone
        results = self.vectorstore.query(query_vector, top_k=top_k)

        return results
