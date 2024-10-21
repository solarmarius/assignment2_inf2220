from typing import List
from langchain.document_loaders import UnstructuredEmailLoader
from langchain.schema import Document


class MyElmLoader:
    def __init__(self, file_path, unstructured_kwargs=None):
        self.file_path = file_path
        self.unstructured_kwargs = unstructured_kwargs or {}

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc
