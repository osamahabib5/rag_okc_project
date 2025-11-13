rag_okc_project

End-to-end Retrieval-Augmented Generation (RAG) Pipeline for NBA Stats

🎯 Project Overview

This project builds a full pipeline that lets you ask natural language questions about NBA game data and get answers grounded in the data. It consists of:

A backend ingestion, embedding, retrieval & answer-generation engine

A frontend chat interface to ask questions interactively

Deployment ready via Docker and Docker Compose

The use case: You have CSVs of NBA game data (2023-24 & 2024-25 seasons, Western Conference teams) → store in PostgreSQL → create embeddings → retrieve relevant rows with semantic search (via pgvector) → feed context + user question into an LLM → get an answer with evidence from the retrieved data.

🧱 Architecture & Key Components
1. Data Ingestion

backend/ingest.py: loads CSV files into PostgreSQL tables (game summaries, box scores, etc.)

Uses Docker-hosted database service (via docker-compose.yml)

2. Embeddings

backend/embed.py: reads rows from the database, serialises text, generates embeddings via nomic‑embed‑text in Ollama, stores embeddings alongside source rows (via pgvector extension)

3. Retrieval & Answer Generation

backend/rag.py: given a question, uses the embedding store + semantic retrieval to find relevant context rows → joins them to structured data rows → constructs a prompt and calls an LLM (e.g., Llama3.2:3b) to produce an answer grounded in the evidence.

Each answer output includes an evidence array listing the game_details or player_box_scores rows used.

4. Frontend Chat Interface

Located under frontend/: built with Angular (v15)

Offers a chat UI that calls the backend server (via REST/HTTP) to send user questions and display answers + evidence

Runs locally at http://localhost:4200 by default when started

5. Deployment via Docker

docker-compose.yml defines services: database (db), Ollama model host (ollama), application (app)

Dockerfile builds the app image

Quick start commands provided in the next section

🚀 Quick Start
Prerequisites

Docker Desktop installed and running (ensures Docker daemon is active)

Node.js 16.x (for frontend)

Backend Setup
git clone https://github.com/osamahabib5/rag_okc_project.git
cd rag_okc_project
# Start DB and model service
docker compose up -d db ollama
# Pull models in Ollama
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.2:3b
# Build the app service
docker compose build app

Running the Pipeline

Ingest data

docker compose run --rm app python -m backend.ingest


Embed data

docker compose run --rm app python -m backend.embed


Note: This step may take significant time depending on hardware.

Run RAG script (answer the pre-set questions)

docker compose run --rm app python -m backend.rag

Launching Frontend & API Server
# Start backend API server
docker compose run --rm --service-ports app uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload

# Then, in a new terminal:
cd frontend
npm install --force
npm start
# Visit http://localhost:4200 in your browser

📂 Project Structure
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

📋 Submission Requirements

Part 1 (Backend RAG): Run the pipeline and answer the 10 game-level prompts in part1/questions.json. Save results in answers.json, each with evidence.

Part 2 (Frontend): Provide a chat interface that interacts with the backend retrieval+LLM pipeline.

Part 3 (Write-up): In part3/responses.txt, address the questions on your approach, learning, and exploratory ideas (each ≤500 words).

Part 4 (Optional): Fine-tune an embedding model (e.g., intfloat/e5-base-v2) on <20 question–context pairs, evaluate retrieval performance, document in part4/responses.txt.

🧠 Why This Matters

This project combines structured sports data, semantic embeddings, vector-search retrieval, and LLMs to create a user-facing application. It demonstrates skills in:

Data ingestion and relational database design

Embedding generation and vector search (via pgvector)

Prompt engineering and LLM grounding in factual data

Full stack development (backend API + frontend chat UI)

DevOps / containerisation with Docker

(Optional) Embedding fine-tuning and retrieval evaluation

🔧 Customisation & Next Steps

Change or extend the dataset (e.g., include Eastern Conference, other seasons)

Use a different embedding model or vector store

Replace Llama3.2 with a larger/smaller LLM depending on resources

Improve frontend UI/UX: add chat history, user authentication, interactive visualisations

Deploy to the cloud (AWS/GCP/Azure) and expose via a web app

Add metrics logging: embedding latency, retrieval recall, user feedback loop

🧾 License & Data Usage

Note: The data provided in this repository is proprietary and strictly confidential. It’s provided exclusively for use within this technical project and must not be copied, shared, or distributed.
If you adapt this architecture with your own publicly available data, ensure that you respect licensing and privacy constraints.
