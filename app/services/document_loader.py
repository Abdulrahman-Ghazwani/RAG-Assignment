import fitz
from docx import Document
from pathlib import Path


class DocumentLoader:
    @staticmethod
    def load_pdf_with_pages(file_path: str):
        pages = []

        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf):
                page_text = page.get_text()

                if page_text and page_text.strip():
                    pages.append({
                        "content": page_text.strip(),
                        "page": page_num + 1
                    })

        return pages

    @staticmethod
    def load_docx_with_pages(file_path: str):
        doc = Document(file_path)

        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        full_text = "\n".join(paragraphs).strip()

        if not full_text:
            return []

        return [
            {
                "content": full_text,
                "page": 1
            }
        ]

    @staticmethod
    def load_file(file_path: str):
        suffix = Path(file_path).suffix.lower()

        if suffix == ".pdf":
            return DocumentLoader.load_pdf_with_pages(file_path)

        if suffix == ".docx":
            return DocumentLoader.load_docx_with_pages(file_path)

        raise ValueError(f"Unsupported file type: {suffix}")