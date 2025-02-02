import os
import logging
from typing import Union, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Inference API", description="API for querying the RAG pipeline")

# Enhanced prompt template that explicitly asks for citations
GENERATOR_TEMPLATE = """
Use the following pieces of context to answer the question. 
Always include specific citations from the source documents to support your answer.
For each key point in your answer, reference the relevant document section using citation numbers (e.g., [1], [2]).
Context: {context}
Question: {question}
Please provide a detailed answer with citations from the source documents.
"""

# Data model for the API request
class QueryRequest(BaseModel):
    query: str

# Data model for the API response
class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, str]]
    confidence_score: float

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Qdrant client and embedding model
def initialize_components():
    """Initialize Qdrant client and embedding model."""
    qdrant_mode = os.getenv("QDRANT_MODE", "local")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_cloud_url = os.getenv("QDRANT_CLOUD_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "legal_documents")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

    # Initialize Qdrant client
    if qdrant_mode == "local":
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
    else:
        client = QdrantClient(url=qdrant_cloud_url, api_key=qdrant_api_key, timeout=60)

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    return client, collection_name, embedding_model

# Initialize components at startup
client, collection_name, embedding_model = initialize_components()

# Helper function to query DeepSeek API
def query_deepseek_api(prompt: str) -> str:
    """Sends a prompt to the DeepSeek API and returns the generated response."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API key not found in environment variables.")
    url = "https://api.deepseek.com/v1/chat/completions"  # Replace with the actual DeepSeek API endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",  # Replace with the correct model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Failed to query DeepSeek API: {e}")
        return "An error occurred while generating the response."


# Define the Citation and QueryResponse models
class Citation(BaseModel):
    text: str
    source: str
    page: Union[str, int]  # Allow both string and integer

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_score: float

# Helper function to format citations
def format_citations(source_documents: List[dict]) -> List[Dict[str, str]]:
    """Formats source documents into structured citations."""
    citations = []
    for idx, doc in enumerate(source_documents):
        citation = {
            'text': doc["payload"]["text"][:1000] + '...' if len(doc["payload"]["text"]) > 1000 else doc["payload"]["text"],
            'source': doc["payload"]["metadata"].get('source', 'Unknown'),
            'page': str(doc["payload"]["metadata"].get('page', 'N/A'))  # Convert page to string
        }
        citations.append(citation)
    return citations

# API endpoint to handle queries

# Update the query_rag_pipeline function
@app.post("/query", response_model=QueryResponse)
async def query_rag_pipeline(request: QueryRequest):
    """Endpoint to query the RAG pipeline."""
    try:
        query = request.query
        logger.info(f"Received query: {query}")

        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)
        logger.info("Generated query embedding.")

        # Search Qdrant for relevant documents
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=int(os.getenv("TOP_K", 4))  # Number of results to retrieve
        )
        logger.info(f"Found {len(search_result)} relevant documents.")

        # Extract source documents from search results
        source_docs = [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in search_result
        ]

        # Format the context from retrieved documents with citation numbers
        context_with_citations = []
        for idx, doc in enumerate(source_docs, start=1):
            snippet = doc["payload"]["text"][:200] + '...' if len(doc["payload"]["text"]) > 200 else doc["payload"]["text"]
            context_with_citations.append(f"[{idx}] {snippet}")
        context = "\n\n".join(context_with_citations)

        # Format the prompt for DeepSeek
        prompt = GENERATOR_TEMPLATE.format(context=context, question=query)
        logger.info("Formatted prompt for DeepSeek API.")

        # Query DeepSeek API
        answer = query_deepseek_api(prompt)
        logger.info("Received response from DeepSeek API.")

        # Format citations from source documents
        citations = format_citations(source_docs)

        # Calculate a simple confidence score based on citation count
        confidence_score = min(len(citations) / 5.0, 1.0)

        # Return the response
        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence_score=confidence_score
        )
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your query: {str(e)}")
# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Inference API!"}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)