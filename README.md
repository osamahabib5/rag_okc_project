# rag_okc_project  
**End-to-End Retrieval-Augmented Generation (RAG) Pipeline for NBA Stats**

---

## 🎯 Project Overview  
This project builds a full pipeline that lets you ask natural language questions about NBA game data and get answers grounded in that data.  

It consists of:  
- A backend for ingestion, embedding, retrieval, and answer generation  
- A frontend chat interface for user interaction  
- Deployment setup with Docker and Docker Compose  

**Use Case:**  
You have CSVs of NBA game data (2023-24 & 2024-25 seasons, Western Conference teams) → store in PostgreSQL → create embeddings → retrieve relevant rows with semantic search (`pgvector`) → feed context + user question into an LLM → get an answer with evidence from the retrieved data.  

---

## 🧱 Architecture & Key Components  

### 1. Data Ingestion  
- **File:** `backend/ingest.py`  
- Loads CSV files into PostgreSQL tables (game summaries, box scores, etc.)  
- Uses Docker-hosted PostgreSQL defined in `docker-compose.yml`  

### 2. Embeddings  
- **File:** `backend/embed.py`  
- Reads rows from the database  
- Generates embeddings using `nomic-embed-text` via Ollama  
- Stores embeddings in PostgreSQL using the `pgvector` extension  

### 3. Retrieval & Answer Generation  
- **File:** `backend/rag.py`  
- Given a question:  
  - Uses semantic retrieval to find relevant context rows  
  - Builds a prompt combining the user question and retrieved data  
  - Calls an LLM (e.g., `llama3.2:3b`) via Ollama  
  - Outputs an answer with an `evidence` list showing which rows were used  

### 4. Frontend Chat Interface  
- **Folder:** `frontend/`  
- Built with Angular  
- Provides a simple chat UI that communicates with the backend API  
- Default local server: [http://localhost:4200](http://localhost:4200)  

### 5. Deployment via Docker  
- **Files:**  
  - `docker-compose.yml` – defines database, model, and app services  
  - `Dockerfile` – builds the app image  

---

## 🚀 Quick Start  

### Prerequisites  
- Docker Desktop installed and running  
- Node.js 16.x or higher (for the frontend)  

### Backend Setup  
```bash
git clone https://github.com/osamahabib5/rag_okc_project.git
cd rag_okc_project

# Start database and model containers
docker compose up -d db ollama

# Pull required models into Ollama
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.2:3b

# Build the app container
docker compose build app
