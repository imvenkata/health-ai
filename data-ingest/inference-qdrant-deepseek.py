import os
import logging
import nltk
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests  # Import requests for API calls
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings  # Import HuggingFaceEmbeddings
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables from .env file
load_dotenv()

# Enhanced prompt template that explicitly asks for citations
GENERATOR_TEMPLATE = """
Use the following pieces of context to answer the question. 
Always include specific citations from the source documents to support your answer.
For each key point in your answer, reference the relevant document section using citation numbers (e.g., [1], [2]).
Context: {context}
Question: {question}
Please provide a detailed answer with citations from the source documents.
"""

@dataclass
class CitedAnswer:
    """Data class to store the answer and its supporting citations."""
    answer: str
    citations: List[Dict[str, str]]
    confidence_score: float


def format_citations(source_documents: List[dict]) -> List[Dict[str, str]]:
    """
    Formats source documents into structured citations.
    
    Args:
        source_documents: List of source documents from the retriever
        
    Returns:
        List of citation dictionaries containing document metadata
    """
    citations = []
    for idx, doc in enumerate(source_documents):
        citation = {
            'text': doc["payload"]["text"][:1000] + '...' if len(doc["payload"]["text"]) > 1000 else doc["payload"]["text"],
            'source': doc["payload"]["metadata"].get('source', 'Unknown'),
            'page': doc["payload"]["metadata"].get('page', 'N/A')  # Default to 'N/A' if page is missing
        }
        citations.append(citation)
    return citations


def query_deepseek_api(prompt: str) -> str:
    """
    Sends a prompt to the DeepSeek API and returns the generated response.
    
    Args:
        prompt: The input prompt to send to the DeepSeek API.
        
    Returns:
        The generated response from the DeepSeek API.
    """
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
        logging.error(f"Failed to query DeepSeek API: {e}")
        return "An error occurred while generating the response."


def initialize_qdrant_pipeline(qdrant_mode: str, qdrant_url: str, qdrant_cloud_url: str, qdrant_api_key: str, collection_name: str):
    """
    Initializes the RAG pipeline components using Qdrant.
    
    Args:
        qdrant_mode: Mode of Qdrant ('local' or 'cloud').
        qdrant_url: URL of the local Qdrant instance.
        qdrant_cloud_url: URL of the cloud Qdrant instance.
        qdrant_api_key: API key for Qdrant authentication.
        collection_name: Name of the collection in Qdrant.
    
    Returns:
        QdrantClient: Initialized Qdrant client.
    """
    logger = logging.getLogger(__name__)
    try:
        # Initialize Qdrant client based on mode
        if qdrant_mode == "local":
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60  # Increase timeout for large queries
            )
        else:
            client = QdrantClient(
                url=qdrant_cloud_url,
                api_key=qdrant_api_key,
                timeout=60  # Increase timeout for large queries
            )
        logger.info("Initialized Qdrant client successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant pipeline: {e}")
        return None


def query_rag_pipeline(client: QdrantClient, collection_name: str, embedding_model, query: str) -> CitedAnswer:
    """
    Queries the RAG pipeline and returns a cited answer.
    
    Args:
        client: The initialized Qdrant client.
        collection_name: Name of the collection in Qdrant.
        embedding_model: Embedding model for generating query embeddings.
        query: The user's question.
        
    Returns:
        CitedAnswer: Contains the answer, supporting citations, and confidence score.
    """
    try:
        if client is None:
            return CitedAnswer(
                answer="RAG pipeline is not initialized.",
                citations=[],
                confidence_score=0.0
            )
        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)  # Use embed_query instead of get_text_embedding
        # Search Qdrant for relevant documents
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=int(os.getenv("TOP_K", 4))  # Number of results to retrieve
        )
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
        # Query DeepSeek API
        answer = query_deepseek_api(prompt)
        # Format citations from source documents
        citations = format_citations(source_docs)
        # Calculate a simple confidence score based on citation count
        confidence_score = min(len(citations) / 5.0, 1.0)
        return CitedAnswer(
            answer=answer,
            citations=citations,
            confidence_score=confidence_score
        )
    except Exception as e:
        logging.error(f"Failed to query RAG pipeline: {e}")
        return CitedAnswer(
            answer="An error occurred while processing your query.",
            citations=[],
            confidence_score=0.0
        )


def print_cited_answer(cited_answer: CitedAnswer):
    """
    Prints the answer with formatted citations.
    
    Args:
        cited_answer: CitedAnswer object containing the answer and citations
    """
    print("\nAnswer:")
    print("=" * 80)
    print(cited_answer.answer)
    print("\nConfidence Score: {:.1%}".format(cited_answer.confidence_score))
    
    if cited_answer.citations:
        print("\nSupporting Citations:")
        print("-" * 80)
        for idx, citation in enumerate(cited_answer.citations, 1):
            print(f"\n[{idx}] Source: {citation['source']}, Page: {citation['page']}")
            print(f"Relevant text: {citation['text']}")


def print_help():
    """Prints help information and example queries."""
    print("\nAvailable Commands:")
    print("- 'help' or '?': Show this help message")
    print("- 'quit', 'exit', or 'q': Exit the program")
    
    print("\nExample Queries:")
    print("- What are the payment terms in the contract?")
    print("- What are the obligations of the Advisor?")
    print("- What is the termination policy?")


def main():
    """Main entry point for the inference pipeline."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize the RAG pipeline
    qdrant_mode = os.getenv("QDRANT_MODE", "local")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_cloud_url = os.getenv("QDRANT_CLOUD_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "legal_documents")
    
    # Initialize Qdrant client
    client = initialize_qdrant_pipeline(qdrant_mode, qdrant_url, qdrant_cloud_url, qdrant_api_key, collection_name)
    
    if client is None:
        logger.error("Failed to initialize RAG pipeline. Exiting...")
        return
    
    # Initialize embedding model
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("\nContract Analysis System")
    print("=" * 80)
    print("Type 'help' or '?' for available commands and example queries.")
    print("=" * 80)
    
    # Main input loop
    while True:
        try:
            query = input("\nEnter your query about the contract: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting the program...")
                break
            elif query.lower() in ['help', '?']:
                print_help()
                continue
            elif not query:
                print("\nPlease enter a valid query.")
                continue
                
            print("\nProcessing your query...\n")
            cited_answer = query_rag_pipeline(client, collection_name, embedding_model, query)
            print_cited_answer(cited_answer)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()