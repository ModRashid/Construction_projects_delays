import re
import fitz

def extract_pdf_pages(pdf_bytes):
    """Extract text page-by-page while preserving page numbers."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text") or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if text:
            pages.append({"page": page_number, "text": text})

    return pages

def build_chunks(pages, chunk_size=1200, overlap=200):
    """Create overlapping chunks and preserve page metadata."""
    records = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()

            if chunk:
                records.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": chunk,
                })
                chunk_id += 1

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return records
