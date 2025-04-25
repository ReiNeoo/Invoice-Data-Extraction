import json
import fitz

from src.extract_data import ExtractData


class ExtractDataFromFile(ExtractData):
    def __init__(self):
        super().__init__(),
        pass

    def get_pages(self, pdf_file: str) -> None:
        doc = fitz.open(pdf_file)
        result = [page.get_text() for page in doc]
        return result
