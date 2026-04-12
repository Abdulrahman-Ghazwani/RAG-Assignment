import fitz  # PyMuPDF
from typing import List


class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str) -> str:
        text_pages: List[str] = []

        try:
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    page_text = page.get_text()
                    if page_text and page_text.strip():
                        text_pages.append(page_text.strip())

        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")

        return "\n\n".join(text_pages)