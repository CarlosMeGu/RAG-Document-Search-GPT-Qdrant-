import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException

from langchain_openai import OpenAIEmbeddings
from document_loader import DocumentLoader
from qdrant_service import QdrantService, UpsertVectorsParams, CreateCollectionParams

logger = logging.getLogger(__name__)


class DocumentIndexingService:
    """Service for processing and indexing documents in vector database."""

    def __init__(
            self,
            collection_name: str = "documents",
            vector_size: int = 1536,
            temp_dir: str = "temp_files"
    ):
        """
        Initialize the document indexing service.

        Args:
            collection_name: Name of the Qdrant collection
            vector_size: Size of embedding vectors
            temp_dir: Directory to store temporary files
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.doc_loader = DocumentLoader()
        self.qdrant_service = QdrantService()
        self.embedding_model = OpenAIEmbeddings()

        # Ensure collection exists
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Initialize Qdrant collection if it doesn't exist."""
        self.qdrant_service.create_collection(CreateCollectionParams(
            collection_name=self.collection_name,
            vector_size=self.vector_size
        ))
        logger.info(f"Ensured Qdrant collection '{self.collection_name}' exists")

    async def save_uploaded_file(self, file: UploadFile) -> Path:
        """
        Save an uploaded file to the temporary directory.

        Args:
            file: The uploaded file

        Returns:
            Path to the saved file
        """
        save_path = self.temp_dir / file.filename

        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File '{file.filename}' saved at '{save_path}'")
        return save_path

    async def extract_document_chunks(self, file_path: Path) -> List[str]:
        """
        Process a file and extract text chunks.

        Args:
            file_path: Path to the file to process

        Returns:
            List of text chunks extracted from the document

        Raises:
            HTTPException: If no valid content could be extracted
        """
        document_texts: List[str] = []

        async for document in self.doc_loader.alazy_load(str(file_path)):
            if document.page_content:
                document_texts.append(document.page_content)
                logger.debug(f"Loaded document chunk (length: {len(document.page_content)}) from '{file_path.name}'")

        if not document_texts:
            logger.error(f"No valid content extracted from '{file_path.name}'")
            raise HTTPException(status_code=400, detail="No valid content extracted from document.")

        logger.info(f"Extracted {len(document_texts)} document chunks from '{file_path.name}'")
        return document_texts

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.embed_documents(texts, self.vector_size)
        logger.info(f"Generated {len(embeddings)} embeddings with vector size {self.vector_size}")
        return embeddings

    def index_in_qdrant(
            self,
            embeddings: List[List[float]],
            texts: List[str],
            filename: str
    ) -> int:
        """
        Index document embeddings in Qdrant.

        Args:
            embeddings: List of embedding vectors
            texts: List of text chunks
            filename: Name of the original file

        Returns:
            Number of chunks indexed
        """
        # Prepare payloads for Qdrant insertion
        payloads: List[Dict[str, str]] = [
            {"filename": filename, "text": text} for text in texts
        ]

        # Create upsert parameters and index the document embeddings
        upsert_params = UpsertVectorsParams(
            collection_name=self.collection_name,
            vectors=embeddings,
            payloads=payloads
        )

        self.qdrant_service.upsert_vectors(upsert_params)
        logger.info(f"Successfully indexed {len(embeddings)} chunks from '{filename}' in Qdrant")
        return len(embeddings)

    async def process_and_index_document(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process and index a document file.

        This method:
        1. Saves the uploaded file
        2. Extracts text chunks from the document
        3. Generates embeddings for each chunk
        4. Indexes the embeddings in Qdrant

        Args:
            file: The uploaded file

        Returns:
            Information about the indexed document

        Raises:
            HTTPException: If processing fails
        """
        file_path = None
        try:
            # Save the uploaded file
            file_path = await self.save_uploaded_file(file)

            # Extract document chunks
            document_chunks = await self.extract_document_chunks(file_path)

            # Generate embeddings
            embeddings = self.generate_embeddings(document_chunks)

            # Index in Qdrant
            chunks_indexed = self.index_in_qdrant(embeddings, document_chunks, file.filename)

            return {
                "filename": file.filename,
                "status": "indexed",
                "chunks": chunks_indexed
            }

        except Exception as e:
            logger.exception(f"Error processing file '{file.filename}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        finally:
            # Clean up temporary files if needed
            # Uncomment the following line if you want to delete files after processing
            # if file_path: file_path.unlink(missing_ok=True)
            pass