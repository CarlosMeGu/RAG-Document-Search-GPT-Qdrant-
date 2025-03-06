import os
import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

# Import your QdrantService and related models
from qdrant_service import QdrantService, SearchVectorsParams

# Use the updated langchain-community imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class TestParameter(BaseModel):
    """Model for test parameters with their reference ranges and units."""
    name: str
    user_value: Optional[float] = None
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    units: Optional[str] = None


class TestData(BaseModel):
    """Model for test data including name and parameters."""
    test_name: str
    parameters: List[TestParameter] = []
    error: Optional[str] = None


# ---------------------------------------------------------------------------------
# 1. The Retriever class: purely for searching vectors and returning contexts
# ---------------------------------------------------------------------------------
class DocumentRetriever:
    """
    DocumentRetriever class responsible for all low-level retrieval
    (vector search, context extraction) from the Qdrant collection.
    """

    def __init__(self,
                 qdrant_service: QdrantService,
                 collection_name: str,
                 embedding_model: str = "text-embedding-ada-002",
                 llm_model: str = "gpt-4o-mini"):
        self.qdrant_service = qdrant_service
        self.collection_name = collection_name

        # Initialize the embedding model
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)

        # Initialize an LLM instance if you want to handle direct user Q&A in this class
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)

    def embed_query(self, query_text: str) -> List[float]:
        """Generate embedding vector for the query text."""
        return self.embedding_model.embed_query(query_text)

    def search_vectors(self, query_vector: List[float], limit: int) -> List[Any]:
        """Search for similar vectors in Qdrant collection."""
        search_params = SearchVectorsParams(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return self.qdrant_service.search_vectors(search_params)

    def extract_contexts(self, search_results: List[Any]) -> List[str]:
        """Extract text contexts from Qdrant search results."""
        contexts = []
        for record in search_results:
            # Handle records with a 'payload' attribute
            if hasattr(record, 'payload') and record.payload:
                text = record.payload.get("text", "")
                if text:
                    contexts.append(text)
            # Handle tuples (if your Qdrant returns them this way)
            elif isinstance(record, tuple):
                if len(record) > 1 and isinstance(record[1], list):
                    for scored_point in record[1]:
                        if hasattr(scored_point, 'payload') and scored_point.payload:
                            text = scored_point.payload.get("text", "")
                            if text:
                                contexts.append(text)
        return contexts

    def retrieve_contexts(self, query_text: str, limit: int = 5) -> List[str]:
        """
        Generic method to retrieve document contexts from Qdrant for a given query.
        """
        # Embed the query
        query_vector = self.embed_query(query_text)
        # Search in Qdrant
        search_results = self.search_vectors(query_vector, limit)
        # Extract the contexts
        return self.extract_contexts(search_results)

    def answer_user_query(self, query_text: str, limit: int = 5) -> Dict[str, str]:
        """
        Answers a user query using ONLY the context found in the documents.
        If the question cannot be answered from the context, respond accordingly.
        """
        # 1) Retrieve context related to the user query
        contexts = self.retrieve_contexts(query_text, limit=limit)
        context_str = "\n\n".join(contexts)

        # 2) If no relevant context is found, return a "cannot answer" response
        if not context_str.strip():
            return {"answer": "I'm not able to answer that question from the document."}

        # 3) Build the LLM prompt
        #    - Instruct the LLM to use ONLY the context to answer
        #    - If it can't answer from context, say so
        prompt = (
            "You have the following context from the document:\n\n"
            f"{context_str}\n\n"
            "Respond to the user's question using ONLY the context provided.\n"
            "Do NOT invent or infer details that aren't in the text.\n"
            "If the question is not related to the context or cannot be answered from it,\n"
            "return: 'I'm not able to answer that question from the document.'\n\n"
            f"User's question:\n{query_text}\n\n"
            "Answer:"
        )

        # 4) Call the LLM and return the answer
        response = self.llm(prompt)
        return {"answer": response.content}




# ------------------- EXAMPLE USAGE -------------------
if __name__ == "__main__":
    # 1) Initialize your QdrantService (ensure QDRANT_API_KEY and QDRANT_HOST are set)
    qdrant_service = QdrantService()

    # 2) Create the retriever
    retriever = DocumentRetriever(
        qdrant_service=qdrant_service,
        collection_name="documents",
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-4o-mini"
    )


    def run_examples():
        print("\n=== ASKING A USER QUESTION ===")
        user_question = "What is the normal range for hemoglobin based on the document?"
        answer_response = retriever.answer_user_query(user_question, limit=10)
        print("Answer to user question:\n", answer_response["answer"])



    run_examples()
