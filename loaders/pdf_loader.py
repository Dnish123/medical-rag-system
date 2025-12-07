"""
Streaming PDF loader for large medical textbooks.
Handles PDFs up to 20,000 pages without loading entire file into memory.
"""

import fitz  # PyMuPDF
from typing import List, Dict
from pathlib import Path
import gc


class StreamingPDFLoader:
    """
    Streams PDF pages one at a time to avoid memory overload.
    """

    def __init__(self, max_size_mb: int = 500):
        """
        Args:
            max_size_mb: Maximum PDF size in MB
        """
        self.max_size_mb = max_size_mb

    def load_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Load PDF and extract text page by page.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page dictionaries with text and metadata
        """
        pdf_path = Path(pdf_path)

        # Validate file
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            raise ValueError(f"PDF too large: {file_size_mb:.1f}MB (max: {self.max_size_mb}MB)")

        pages = []

        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            print(f"Processing {total_pages} pages...")

            # Stream pages
            for page_num in range(total_pages):
                page = doc[page_num]

                # Extract text
                text = page.get_text()

                # Skip empty pages
                if not text.strip():
                    continue

                # Create page object
                pages.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'char_count': len(text)
                })

                # Periodic garbage collection for large PDFs
                if page_num % 100 == 0:
                    gc.collect()

            doc.close()

        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")

        return pages

    def extract_metadata(self, pdf_path: str) -> Dict:
        """Extract PDF metadata"""
        doc = fitz.open(pdf_path)
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'pages': len(doc),
            'file_size': Path(pdf_path).stat().st_size
        }
        doc.close()
        return metadata
