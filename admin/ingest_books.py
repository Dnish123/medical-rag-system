"""
ADMIN-ONLY SCRIPT
Run this script once to ingest medical books into Pinecone.
This should NOT be accessible to regular users.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from loaders.pdf_loader import StreamingPDFLoader
from embeddings.embedder import ChunkEmbedder
from vectorstore.pinecone_db import PineconeDB
from utils.config import Config
import argparse
from tqdm import tqdm


def ingest_book(pdf_path: str, book_name: str):
    """
    Ingest a single medical book into Pinecone.

    Args:
        pdf_path: Path to PDF file
        book_name: Name of the book for metadata
    """
    print(f"\n🔄 Starting ingestion for: {book_name}")
    print(f"📄 PDF: {pdf_path}")

    # Initialize components
    loader = StreamingPDFLoader()
    embedder = ChunkEmbedder()
    vectorstore = PineconeDB()

    try:
        # Step 1: Load and stream PDF
        print("\n📖 Loading PDF...")
        pages = loader.load_pdf(pdf_path)
        print(f"✅ Loaded {len(pages)} pages")

        # Step 2: Create chunks
        print("\n✂️ Creating chunks...")
        chunks = embedder.create_chunks(pages, book_name)
        print(f"✅ Created {len(chunks)} chunks")

        # Step 3: Generate embeddings in batches
        print("\n🧮 Generating embeddings...")
        all_vectors = []

        batch_size = Config.BATCH_SIZE
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
            batch = chunks[i:i + batch_size]
            vectors = embedder.embed_batch(batch)
            all_vectors.extend(vectors)

        print(f"✅ Generated {len(all_vectors)} embeddings")

        # Step 4: Upload to Pinecone
        print("\n☁️ Uploading to Pinecone...")
        vectorstore.upsert_vectors(all_vectors)
        print("✅ Upload complete!")

        # Step 5: Verify
        stats = vectorstore.get_index_stats()
        print(f"\n📊 Index stats: {stats['total_vector_count']} vectors")

        print(f"\n🎉 Successfully ingested {book_name}!")

    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Ingest medical books into Pinecone")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--name", required=True, help="Book name for metadata")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")

    args = parser.parse_args()

    # Check if index exists
    vectorstore = PineconeDB()

    if args.rebuild:
        print("⚠️ Rebuilding index from scratch...")
        response = input("This will delete all existing vectors. Continue? (yes/no): ")
        if response.lower() == 'yes':
            vectorstore.delete_all()
            print("✅ Index cleared")
        else:
            print("❌ Cancelled")
            return

    # Ingest book
    ingest_book(args.pdf, args.name)


if __name__ == "__main__":
    main()
"""bash
python admin/ingest_books.py --pdf /path/to/surgery.pdf --name "Bailey Surgery 27th Edition"
"""