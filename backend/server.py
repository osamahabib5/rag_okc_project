# server.py (UPDATED)

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
    extract_score_from_question,
    extract_exact_points,
    retrieve_games,
    retrieve_player_stats,
    filter_by_dates,
    filter_by_player,
    filter_by_matchup,
    filter_by_teams,
    filter_by_opponent,
    filter_by_score,
    filter_by_exact_points,
    find_leading_scorer,
    build_game_context,
    build_player_context,
    extract_structured_answer,
    validate_game_context,
    find_opponent_team_from_db,
    get_player_stats_directly
)

app = FastAPI(title="NBA Stats RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],
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
    return {
        "message": "NBA Stats RAG API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat"
        }
    }


@app.get("/api/health")
def health():
    """Health check endpoint"""
    try:
        with eng.connect() as cx:
            result = cx.execute(text("SELECT COUNT(*) FROM game_details")).scalar()
            player_count = cx.execute(text("SELECT COUNT(*) FROM player_box_scores")).scalar()
        return {
            "status": "healthy",
            "database": "connected",
            "stats": {
                "games": result,
                "player_performances": player_count
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "database": "disconnected"
        }


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
        exact_points = extract_exact_points(question)
        player_name = extract_player_name(question)
        all_teams = extract_all_teams_from_question(question)
        asked_team = extract_team_from_question(question)
        opponent_team = extract_opponent_team(question)
        score1, score2 = extract_score_from_question(question)
        
        debug_info = {
            "query_type": query_type,
            "dates": special_dates,
            "exact_points": exact_points,
            "player_name": player_name,
            "all_teams": all_teams,
            "asked_team": asked_team,
            "opponent_team": opponent_team,
            "score": f"{score1}-{score2}" if score1 and score2 else None
        }
        
        print(f"Debug info: {debug_info}")
        
        # Generate embedding
        qvec = ollama_embed(EMBED_MODEL, question)
        
        with eng.begin() as cx:
            # SPECIAL LOGIC: Player + Date + One Team
            if query_type == 'player' and player_name and special_dates and len(all_teams) == 1 and not exact_points:
                print(f"🔍 Special case: Player + Date + One Team")
                mentioned_team = all_teams[0]
                other_team, game_id = find_opponent_team_from_db(cx, mentioned_team, special_dates, player_name)
                
                if other_team and game_id:
                    rows = get_player_stats_directly(cx, player_name, game_id)
                    
                    if rows:
                        ctx = build_player_context(rows, question)
                        prompt = f"""Answer using ONLY the context below. Use the FIRST entry.

Context:
{ctx}

Question: {question}

Instructions:
- Extract: PLAYER name, PTS, REB, AST
- Use EXACT numbers
- Format: "[Player] scored [X] points with [Y] rebounds and [Z] assists"

Answer:"""
                        
                        llm_response = ollama_generate(LLM_MODEL, prompt)
                        structured_result = extract_structured_answer(llm_response, None, query_type, rows, question)
                        
                        evidence = [{"table": "player_box_score", "id": int(rows[0]['game_id'])}] if rows else []
                        
                        return ChatResponse(
                            answer=llm_response,
                            structured_result=structured_result,
                            evidence=evidence,
                            debug_info=debug_info
                        )
                
                # If special case didn't work, fall through to regular logic
            
            # REGULAR LOGIC
            if query_type == 'player':
                rows = retrieve_player_stats(cx, qvec, k=20)
                
                # Apply filters
                if special_dates:
                    rows = filter_by_dates(rows, special_dates)
                    print(f"After date filter: {len(rows)} rows")
                
                if all_teams and len(all_teams) >= 2:
                    rows = filter_by_matchup(rows, all_teams)
                    print(f"After matchup filter: {len(rows)} rows")
                
                if score1 and score2:
                    rows = filter_by_score(rows, score1, score2)
                    print(f"After score filter: {len(rows)} rows")
                
                if all_teams and len(all_teams) >= 2:
                    if not validate_game_context(rows, all_teams, score1, score2):
                        return ChatResponse(
                            answer="I couldn't find a game matching all the specified criteria. Please verify the date, teams, and score.",
                            structured_result={},
                            evidence=[],
                            debug_info=debug_info
                        )
                
                if player_name:
                    rows = filter_by_player(rows, player_name)
                    print(f"After player filter: {len(rows)} rows")
                
                if exact_points is not None:
                    rows = filter_by_exact_points(rows, exact_points)
                    print(f"After exact points filter: {len(rows)} rows")
                    
                    if not rows:
                        return ChatResponse(
                            answer=f"No player scored exactly {exact_points} points on the specified date.",
                            structured_result={},
                            evidence=[],
                            debug_info=debug_info
                        )
                
                if opponent_team and not all_teams:
                    rows = filter_by_opponent(rows, opponent_team)
                    print(f"After opponent filter: {len(rows)} rows")
                
                if asked_team and not all_teams:
                    rows = filter_by_teams(rows, asked_team)
                    print(f"After team filter: {len(rows)} rows")
                
                if exact_points is None:
                    rows = find_leading_scorer(rows, question)
                
                if not rows:
                    return ChatResponse(
                        answer="I couldn't find any matching player data. Try asking about a different game or player.",
                        structured_result={},
                        evidence=[],
                        debug_info=debug_info
                    )
                
                ctx = build_player_context(rows, question)
                
                prompt = f"""Answer using ONLY the context below. Use the FIRST entry.

Context:
{ctx}

Question: {question}

Instructions:
- The FIRST entry is the answer
- Extract: PLAYER name, PTS, REB, AST
- Use EXACT numbers
- Format: "[Player] scored [X] points with [Y] rebounds and [Z] assists in game_id [ID]"

Answer:"""
                
            else:  # game query
                rows = retrieve_games(cx, qvec, k=8)
                
                if special_dates:
                    rows = filter_by_dates(rows, special_dates)
                    print(f"After date filter: {len(rows)} rows")
                
                if asked_team and opponent_team:
                    rows = filter_by_teams(rows, asked_team, opponent_team)
                    print(f"After team filter: {len(rows)} rows")
                elif asked_team or opponent_team:
                    rows = filter_by_teams(rows, asked_team or opponent_team)
                    print(f"After team filter: {len(rows)} rows")
                
                if not rows:
                    return ChatResponse(
                        answer="I couldn't find any matching game data. Please check the date and team names.",
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
- Match exact date
- Use HOME_PTS for home score, AWAY_PTS for away score
- If ASKED_PTS shown, use that
- Use EXACT numbers
- Format: "[Team] scored [X] points. Winner: [Team] ([HOME/AWAY]). Final: [X-Y]. game_id: [ID]"

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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)