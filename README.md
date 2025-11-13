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
```

### Running the Pipeline

#### Ingest data
```bash
docker compose run --rm app python -m backend.ingest
```

#### Embed data
```bash
docker compose run --rm app python -m backend.embed
```
***Note: This step may take significant time depending on hardware.***

#### Run RAG script (answer the pre-set questions)
```bash
docker compose run --rm app python -m backend.rag
```
#### Launching Frontend & API Server
```bash
# Start backend API server
docker compose run --rm --service-ports app uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload

# Then, in a new terminal:
cd frontend
npm install --force
npm start

# Visit http://localhost:4200 in your browser
```

### Project Structure
```bash
├── backend/
│   ├── ingest.py       # CSV → PostgreSQL
│   ├── embed.py        # Generate & store embeddings
│   ├── rag.py          # Retrieval + LLM answering script
│   └── server.py       # FastAPI/UVicorn backend API (for chat interface)
├── frontend/           # Angular chat interface
├── part1/              # Part 1 assignment assets (questions.json etc)
├── part2/              # Part 2 assets (UI demo video etc)
├── part3/              # Part 3 – write-up responses
├── part4/              # Optional fine-tuning of embeddings
├── prompts/            # Directory to list any AI prompts used
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
## 🎯 Final Output

- Part 1 (Backend RAG): Run the pipeline and answer the 10 game-level prompts in part1/questions.json. Save results in answers.json, each with evidence.
- Part 2 (Frontend): Provide a chat interface that interacts with the backend retrieval+LLM pipeline.

## 🧠 Why This Matters
This project combines structured sports data, semantic embeddings, vector-search retrieval, and LLMs to create a user-facing application. It demonstrates skills in:

- Data ingestion and relational database design
- Embedding generation and vector search (via pgvector)
- Prompt engineering and LLM grounding in factual data
- Full stack development (backend API + frontend chat UI)
- DevOps / containerisation with Docker
- (Optional) Embedding fine-tuning and retrieval evaluation

## 🔧 Customisation & Next Steps

- Change or extend the dataset (e.g., include Eastern Conference, other seasons)
- Use a different embedding model or vector store
- Replace Llama3.2 with a larger/smaller LLM depending on resources
- Improve frontend UI/UX: add chat history, user authentication, interactive visualisations
- Deploy to the cloud (AWS/GCP/Azure) and expose via a web app
- Add metrics logging: embedding latency, retrieval recall, user feedback loop

