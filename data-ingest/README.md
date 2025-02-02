qdrant:

docker run -p 6333:6333 -v /Users/venkata/ai-apps/health-ai/data-ingest/data/qdrant:/qdrant/storage  qdrant/qdrant

http://localhost:6333/dashboard



app.py


curl -X POST "http://localhost:8000/query" \
-H "Content-Type: application/json" \
-d '{"query": "What are the payment terms in the contract?"}'