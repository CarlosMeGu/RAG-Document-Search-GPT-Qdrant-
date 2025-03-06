# Document Processing and Retrieval API

This project is a beginner-friendly introduction to building AI applications that can understand, process, and answer questions about your documents. If you're new to AI development, this is a perfect starting point to learn about embedding models, vector databases, and large language models (LLMs) in a practical application.

## What This Project Does

Imagine uploading any document (PDF, Word, etc.) and being able to ask questions about its contents in plain English. This system:

1. **Reads your documents** - Extracts text from various file formats
2. **Understands the content** - Converts text chunks into AI-friendly numeric representations (embeddings)
3. **Stores information efficiently** - Saves these representations in a special database designed for AI
4. **Answers your questions** - Uses an AI model to find relevant information and generate human-like responses

## AI Concepts for Beginners

Before diving into the code, let's understand some key AI concepts used in this project:

### 1. Embeddings

**What are embeddings?** Think of embeddings as the AI's way of understanding text. They convert words and sentences into lists of numbers (vectors) that capture their meaning.

**Why are they important?** Embeddings allow the AI to understand that "car" and "automobile" are similar concepts, even though they're different words. This project uses OpenAI's text embeddings to understand document content.

### 2. Vector Databases

**What is a vector database?** Unlike traditional databases that search by exact matching, vector databases (like Qdrant in this project) store and search embeddings based on similarity.

**Why use them?** They let you find information based on meaning rather than keywords. Searching for "What's the company revenue?" might return content about "financial performance" or "quarterly earnings" even if those exact words weren't in your query.

### 3. Large Language Models (LLMs)

**What are LLMs?** These are AI models like GPT-4 that can understand and generate human-like text. They've been trained on vast amounts of text from the internet.

**How are they used here?** Our system uses an LLM to generate natural responses to your questions based on information retrieved from your documents.

### 4. Retrieval Augmented Generation (RAG)

**What is RAG?** It's a technique that combines document retrieval with text generation to create responses based on specific information.

**Why is it powerful?** Instead of just generating responses from general knowledge, RAG lets the AI answer questions based on your specific documents.

## Step-by-Step Guide

### Setting Up Your First AI Document System

#### Prerequisites

- Basic Python knowledge
- Python 3.8 or higher installed
- Access to these API services (free tiers available):
  - [OpenAI](https://platform.openai.com/signup) - For embeddings and language model
  - [Qdrant Cloud](https://cloud.qdrant.io/) - For vector storage
  - [Unstructured](https://unstructured.io/) - For document processing

#### Installation

1. **Clone the project** (or download and extract the ZIP):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up a virtual environment** (keeps your project dependencies organized):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** with your API keys:
   ```
   # Qdrant Configuration
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_HOST=your_qdrant_host_url

   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key

   # Unstructured API Configuration
   UNSTRUCTURED_API_KEY=your_unstructured_api_key
   UNSTRUCTURED_API_URL=your_unstructured_api_url
   ```

#### Running Your AI Application

Start the application with:

```bash
python main.py
```

This will launch your AI application at `http://localhost:8000`. You can then:

1. Open `http://localhost:8000/docs` in your browser to see the interactive API documentation
2. Upload documents using the `/upload/` endpoint
3. Ask questions about your documents using the `/query/` endpoint

## How It Works: The AI Pipeline Explained

Here's what happens when you interact with the system:

### When You Upload a Document:

1. **Document Processing**: The document is sent to the Unstructured API, which extracts text content
2. **Chunking**: Long documents are broken into smaller, manageable pieces
3. **Embedding Generation**: Each chunk is converted into an embedding vector using OpenAI's model
4. **Vector Storage**: These embeddings are stored in Qdrant along with the original text

### When You Ask a Question:

1. **Query Embedding**: Your question is converted into an embedding
2. **Similarity Search**: The system finds document chunks with embeddings most similar to your question
3. **Context Building**: The most relevant chunks are retrieved
4. **Answer Generation**: An LLM (like GPT-4) generates an answer based on the retrieved context

## Understanding the Code

Let's explore the main components:

### `document_loader.py`

This handles the reading and processing of documents:

```python
# Example: Loading and splitting a document
from document_loader import DocumentLoader

loader = DocumentLoader()
document_chunks = loader.load_and_split("your_document.pdf")
```

### `qdrant_service.py`

This manages the vector database operations:

```python
# Example: Creating a collection for document embeddings
from qdrant_service import QdrantService, CreateCollectionParams

qdrant = QdrantService()
qdrant.create_collection(CreateCollectionParams(
    collection_name="my_documents",
    vector_size=1536,  # Size of OpenAI embeddings
    distance_metric="COSINE"
))
```

### `document_retriever.py`

This handles retrieving relevant document parts and generating answers:

```python
# Example: Answering a question from indexed documents
from document_retriever import DocumentRetriever

retriever = DocumentRetriever(
    qdrant_service=qdrant_service,
    collection_name="my_documents"
)
answer = retriever.answer_user_query("What does the document say about AI safety?")
```

## Exercises for Learning

Try these exercises to build your understanding:

1. **Modify the chunk size**: Change the `chunk_size` parameter in `document_indexing_service.py` to see how it affects retrieval quality
2. **Add metadata filtering**: Enhance the retriever to filter results by document type or date
3. **Implement custom prompts**: Modify the prompt template in `document_retriever.py` to change how the AI responds

## Learning Path for AI Beginners

To deepen your understanding of AI and this project:

1. **Start with embeddings**: Learn how they capture meaning in text
   - Resource: [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)

2. **Explore vector databases**: Understand similarity search
   - Resource: [Qdrant Documentation](https://qdrant.tech/documentation/)

3. **Learn about LLMs**: Understand how they generate text
   - Resource: [OpenAI API Documentation](https://platform.openai.com/docs/guides/text-generation)

4. **Study RAG architectures**: See how retrieval enhances generation
   - Resource: [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)

## Common Challenges and Solutions

**Issue**: My document isn't being processed correctly.
**Solution**: The Unstructured API works best with well-formatted documents. Try different document formats or check for complex layouts.

**Issue**: The answers aren't relevant to my questions.
**Solution**: Try adjusting the `limit` parameter when querying to retrieve more context, or experiment with different chunking strategies.

**Issue**: I'm getting API rate limit errors.
**Solution**: Most services offer free tiers with limitations. Consider implementing caching or throttling to stay within limits.

## Next Steps in Your AI Journey

After mastering this project, consider these extensions:

1. **Add a web interface**: Create a simple frontend for non-technical users
2. **Implement multi-user support**: Add authentication and user-specific collections
3. **Try different embedding models**: Experiment with various models to compare performance
4. **Add document classification**: Automatically categorize uploaded documents

## Need Help?

If you're new to AI and need assistance:

1. Check out the API documentation at `http://localhost:8000/docs`
2. Explore the code comments for explanations
3. Join AI community forums like Hugging Face, LangChain, or Stack Overflow

Welcome to your AI development journey!