"""
Chunking and embedding logic for medical textbooks.
Uses semantic chunking with overlap for better context preservation.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
from utils.config import Config
import numpy as np
import re


class ChunkEmbedder:
    """
    Handles text chunking and embedding generation.
    """

    def __init__(self):
        """Initialize embedding model"""
        print(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP

    def create_chunks(self, pages: List[Dict], book_name: str) -> List[Dict]:
        """
        Create overlapping chunks from pages.

        Args:
            pages: List of page dictionaries
            book_name: Name of the book

        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        chunk_id = 0

        for page in pages:
            page_text = page['text']
            page_num = page['page_number']

            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(page_text)

            for para_num, paragraph in enumerate(paragraphs, 1):
                # Skip very short paragraphs
                if len(paragraph.split()) < 20:
                    continue

                # Create chunks with overlap
                para_chunks = self._chunk_text(paragraph)

                for chunk_text in para_chunks:
                    chunks.append({
                        'id': f"{book_name}_page{page_num}_para{para_num}_chunk{chunk_id}",
                        'text': chunk_text,
                        'metadata': {
                            'book': book_name,
                            'page': page_num,
                            'paragraph': para_num,
                            'chunk_id': chunk_id
                        }
                    })
                    chunk_id += 1

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or multiple spaces
        paragraphs = re.split(r'\n\n+|\n\s+\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with overlap based on token count.
        Uses approximate token count (words * 1.3)
        """
        words = text.split()
        chunks = []

        # Approximate tokens (1 token ≈ 0.75 words)
        words_per_chunk = int(self.chunk_size * 0.75)
        words_overlap = int(self.chunk_overlap * 0.75)

        start = 0
        while start < len(words):
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))

            # Move start by (chunk_size - overlap)
            start += (words_per_chunk - words_overlap)

        return chunks

    def embed_batch(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a batch of chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of vectors ready for Pinecone upload
        """
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Format for Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            vectors.append({
                'id': chunk['id'],
                'values': embeddings[i].tolist(),
                'metadata': {
                    **chunk['metadata'],
                    'text': chunk['text'][:1000]  # Pinecone metadata limit
                }
            })

        return vectors

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

