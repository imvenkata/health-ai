qdrant:

docker run -p 6333:6333 -v /Users/venkata/ai-apps/health-ai/data-ingest/data/qdrant:/qdrant/storage  qdrant/qdrant

http://localhost:6333/dashboard



app.py





curl -X POST "http://localhost:8000/query" \
-H "Content-Type: application/json" \
-d '{"query": "What are the foods to avoid for ADHD"}'



 code2prompt --path /Users/venkata/ai-apps/health-ai/data-ingest  --exclude "*/Users/venkata/ai-apps/health-ai/data-ingest/.venv*"    --output  /Users/venkata/ai-apps/health-ai/project_summary.md

 code2prompt --path /Users/venkata/ai-apps/health-ai/rag-frontend/  --exclude "**.log,**.tmp,**/node_modules/**,**.next**,**public**"    --output  /Users/venkata/ai-apps/health-ai/front_end_summary.md


frontend:
npm init -y
npm install express
npm install axios

node server.js




order to run: 

Qdrant:

docker spinup:

docker desktop 

```bash
docker run -p 6333:6333 -v /Users/venkata/ai-apps/health-ai/data-ingest/data/qdrant:/qdrant/storage  qdrant/qdrant
```

backend:

query engine:

```bash
cd /Users/venkata/ai-apps/health-ai/data-ingest
source .venv/bin/activate
python app.py
```
frontend:

```bash
cd /Users/venkata/ai-apps/health-ai/rag-frontend
npm run dev
```
