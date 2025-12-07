"""
Configuration management for the Medical RAG System.
Loads environment variables and provides validation.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()


class Config:
    """Central configuration class"""

    # API Keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Pinecone Settings
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-rag-aiims")
    PINECONE_DIMENSION = 384  # all-MiniLM-L6-v2 embedding dimension
    PINECONE_METRIC = "cosine"

    # Embedding Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 200
    BATCH_SIZE = 64

    # Retrieval Settings
    TOP_K = 5

    # LLM Settings
    GROQ_MODEL_PRIMARY = "llama-3.3-70b-versatile"
    GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS = 2048
    GROQ_TEMPERATURE = 0.3

    # System Settings
    MAX_PDF_SIZE_MB = 500
    STREAMING_ENABLED = True

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment")
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment")
        return True


# Validate on import
Config.validate()