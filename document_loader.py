import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, AsyncIterator, Iterator

# Load environment variables
load_dotenv()

class DocumentLoader:
    """Handles document loading using the Unstructured API"""

    def __init__(self, api_key: str = None, api_url: str = None):
        self.api_key: str = api_key or os.getenv("UNSTRUCTURED_API_KEY")
        self.api_url: str = api_url or os.getenv("UNSTRUCTURED_API_URL")

        if not self.api_key:
            raise ValueError("Unstructured API key must be provided.")
        if not self.api_url:
            raise ValueError("Unstructured API URL must be provided.")

    def _get_loader(self, file_path: str) -> UnstructuredAPIFileLoader:
        """Creates an instance of UnstructuredAPIFileLoader."""
        return UnstructuredAPIFileLoader(
            file_path=file_path,
            api_key=self.api_key,
            url=self.api_url,
        )

    def load(self, file_path: str) -> List[Document]:
        """Loads the document synchronously.

        Args:
            file_path (str): Path to the document.

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        loader = self._get_loader(file_path)
        return loader.load()

    def lazy_load(self, file_path: str) -> Iterator[Document]:
        """Loads the document lazily (for large files).

        Args:
            file_path (str): Path to the document.

        Returns:
            Iterator[Document]: An iterator yielding LangChain Document objects.
        """
        loader = self._get_loader(file_path)
        return loader.lazy_load()

    async def aload(self, file_path: str) -> List[Document]:
        """Loads the document asynchronously.

        Args:
            file_path (str): Path to the document.

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        loader = self._get_loader(file_path)
        return await loader.aload()

    async def alazy_load(self, file_path: str) -> AsyncIterator[Document]:
        """Loads the document lazily and asynchronously.

        Args:
            file_path (str): Path to the document.

        Returns:
            AsyncIterator[Document]: An async iterator yielding LangChain Document objects.
        """
        loader = self._get_loader(file_path)
        async for doc in loader.alazy_load():
            yield doc

    def load_and_split(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """Loads the document and splits it into chunks.

        Args:
            file_path (str): Path to the document.
            chunk_size (int): Size of each chunk (default: 1000).
            chunk_overlap (int): Overlap between chunks (default: 100).

        Returns:
            List[Document]: A list of chunked LangChain Document objects.
        """
        documents = self.load(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

# Pydantic Model for API Documentation (Optional)
class DocumentLoaderParams(BaseModel):
    """Defines parameters for document loading functions."""
    file_path: str = Field(..., description="Path to the document file")
    chunk_size: int = Field(1000, description="Chunk size for splitting documents")
    chunk_overlap: int = Field(100, description="Overlap size between chunks")

# Example Usage
if __name__ == "__main__":
    file_path = "my_doc.pdf"
    doc_loader = DocumentLoader()

    # Test each method
    print("\n--- Sync Load ---")
    documents = doc_loader.load(file_path)
    for doc in documents:
        print(doc.page_content[:200])

    print("\n--- Lazy Load ---")
    for doc in doc_loader.lazy_load(file_path):
        print(doc.page_content[:200])

    print("\n--- Load and Split ---")
    chunked_docs = doc_loader.load_and_split(file_path)
    for doc in chunked_docs:
        print(doc.page_content[:200])
