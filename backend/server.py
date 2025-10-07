from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy import text
from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate
from backend.rag import (
    parse_special_dates, 
    extract_player_name, 
    determine_query_type,
    extract_all_teams_from_question,
    extract_team_from_question,
    extract_opponent_team,
    extract_points_from_question,
    retrieve_games,
    retrieve_player_stats,
    filter_by_dates,
    filter_by_player,
    filter_by_points,
    filter_by_matchup,
    filter_by_teams,
    filter_by_opponent,
    find_leading_scorer,
    build_game_context,
    build_player_context,
    extract_structured_answer
)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

eng = sa.create_engine(DB_DSN)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    structured_result: dict = None
    evidence: list = []
    debug_info: dict = None


@app.get("/")
def root():
    return {"message": "NBA Stats RAG API is running"}


@app.get("/api/health")
def health():
    """Health check endpoint"""
    try:
        with eng.connect() as cx:
            cx.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint for NBA stats queries"""
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        # Determine query type
        query_type = determine_query_type(question)
        print(f"Query type: {query_type}")
        
        # Extract contextual information
        special_dates = parse_special_dates(question)
        player_name = extract_player_name(question)
        all_teams = extract_all_teams_from_question(question)
        asked_team = extract_team_from_question(question)
        opponent_team = extract_opponent_team(question)
        specific_points = extract_points_from_question(question)
        
        debug_info = {
            "query_type": query_type,
            "dates": special_dates,
            "player_name": player_name,
            "all_teams": all_teams,
            "asked_team": asked_team,
            "opponent_team": opponent_team,
            "specific_points": specific_points
        }
        
        # Generate embedding
        qvec = ollama_embed(EMBED_MODEL, question)
        
        with eng.begin() as cx:
            if query_type == 'player':
                rows = retrieve_player_stats(cx, qvec, k=20)
                
                # Apply filters
                if special_dates:
                    rows = filter_by_dates(rows, special_dates)
                
                if len(all_teams) >= 2:
                    rows = filter_by_matchup(rows, all_teams)
                elif asked_team and opponent_team:
                    rows = filter_by_teams(rows, asked_team, opponent_team)
                elif asked_team:
                    rows = filter_by_teams(rows, asked_team)
                elif opponent_team:
                    rows = filter_by_opponent(rows, opponent_team)
                
                if specific_points:
                    rows = filter_by_points(rows, specific_points)
                
                if player_name:
                    rows = filter_by_player(rows, player_name)
                
                rows = find_leading_scorer(rows, question)
                
                if not rows:
                    return ChatResponse(
                        answer="I couldn't find any matching player data for your question. Please try rephrasing or asking about a different game.",
                        structured_result={},
                        evidence=[],
                        debug_info=debug_info
                    )
                
                ctx = build_player_context(rows, question)
                
                prompt = f"""Answer using ONLY the context below. Use the FIRST entry as the answer.

Context:
{ctx}

Question: {question}

Instructions:
- The FIRST entry in the context is the correct answer
- Extract: PLAYER name, PTS (points), REB (rebounds), AST (assists)
- Use EXACT numbers from the context
- Format: "[Player Name] scored [X] points with [Y] rebounds and [Z] assists in game_id [ID]"

Answer:"""
                
            else:  # game query
                rows = retrieve_games(cx, qvec, k=8)
                
                if special_dates:
                    rows = filter_by_dates(rows, special_dates)
                if asked_team and opponent_team:
                    rows = filter_by_teams(rows, asked_team, opponent_team)
                elif asked_team or opponent_team:
                    rows = filter_by_teams(rows, asked_team or opponent_team)
                
                if not rows:
                    return ChatResponse(
                        answer="I couldn't find any matching game data for your question. Please try rephrasing or asking about a different game.",
                        structured_result={},
                        evidence=[],
                        debug_info=debug_info
                    )
                
                ctx = build_game_context(rows, question)
                
                prompt = f"""Answer using ONLY the context below.

Context:
{ctx}

Question: {question}

Instructions:
- Match the exact date from the question
- Use HOME_PTS for home team score, AWAY_PTS for away team score
- If ASKED_PTS is shown, that's the answer for the asked team
- Use EXACT numbers from the context
- Format: "[Team] scored [X] points. Winner: [Team] ([HOME/AWAY]). Final Score: [X-Y]. game_id: [ID]"

Answer:"""
            
            # Generate response
            llm_response = ollama_generate(LLM_MODEL, prompt)
            
            # Extract structured result
            structured_result = extract_structured_answer(llm_response, None, query_type, rows, question)
            
            # Build evidence
            evidence = []
            if rows:
                if query_type == 'player':
                    evidence = [{"table": "player_box_score", "id": int(rows[0]['game_id'])}]
                else:
                    evidence = [{"table": "game_details", "id": int(rows[0]['game_id'])}]
            
            print(f"Response: {llm_response}")
            print(f"Structured result: {structured_result}")
            
            return ChatResponse(
                answer=llm_response,
                structured_result=structured_result,
                evidence=evidence,
                debug_info=debug_info
            )
    
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)