import logging
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv
from pydantic import BaseModel

from document_indexing_service import DocumentIndexingService
from document_retriever import DocumentRetriever
from qdrant_service import QdrantService


class UserQuery(BaseModel):
    query: str
    limit: int = 5

qdrant_service = QdrantService()

retriever = DocumentRetriever(
    qdrant_service=qdrant_service,
    collection_name="documents",
    embedding_model="text-embedding-ada-002",
    llm_model="gpt-4o-mini"
)
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for processing and indexing documents",
    version="1.0.0"
)

# Initialize document indexing service
document_service = DocumentIndexingService(
    collection_name="documents",
    vector_size=1536,
    temp_dir="temp_files"
)


@app.post("/upload/", summary="Upload and index a document")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a document file for processing and indexing.

    This endpoint:
    - Accepts a document file upload
    - Processes the document to extract text
    - Generates embeddings for the text
    - Stores the embeddings in a vector database for future retrieval

    Args:
        file: The document file to upload and process

    Returns:
        Information about the indexed document including:
        - filename: Name of the processed file
        - status: Processing status
        - chunks: Number of text chunks extracted and indexed
    """
    logger.info(f"Received upload request for file '{file.filename}'")
    result = await document_service.process_and_index_document(file)
    return result


@app.get("/", summary="Root endpoint")
async def root():
    """
    Root endpoint to verify API is running.

    Returns:
        Simple welcome message
    """
    return {"message": "Document Processing API is running"}


# Add the new endpoint for user queries
@app.post("/query/", summary="Answer a user query based on indexed documents")
async def answer_query(query_data: UserQuery) -> Dict[str, Any]:
    """
    Process a user query and return an answer based on the indexed documents.

    Args:
        query_data: Object containing the user's query and optional limit parameter

    Returns:
        The answer to the user's question based on the document contexts
    """
    logger.info(f"Received query request: '{query_data.query}'")

    try:
        result = retriever.answer_user_query(
            query_text=query_data.query,
            limit=query_data.limit
        )
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"error": f"Failed to process query: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)