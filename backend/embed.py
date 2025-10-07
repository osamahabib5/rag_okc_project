"""
Create embeddings – Generate text embeddings with Ollama nomic-embed-text and store them alongside the source rows.
"""

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from backend.config import DB_DSN, EMBED_MODEL
from backend.utils import ollama_embed

def row_text_game(r):
    """Create embedding text for game_details table"""
    ts = pd.to_datetime(r['game_timestamp'], utc=True)
    date = ts.strftime('%Y-%m-%d')
    day_name = ts.strftime('%A')
    month_day = ts.strftime('%B %d')
    
    # Get team names
    home_team = r['home_team_name'] if 'home_team_name' in r else f"team_{r['home_team_id']}"
    away_team = r['away_team_name'] if 'away_team_name' in r else f"team_{r['away_team_id']}"
    
    hp = int(r['home_points'])
    ap = int(r['away_points'])
    
    # Determine winner
    winner = home_team if hp > ap else away_team
    
    # Special date handling
    special_date = ""
    if ts.month == 12 and ts.day == 25:
        special_date = "Christmas Day | Christmas | "
    elif ts.month == 12 and ts.day == 24:
        special_date = "Christmas Eve | "
    elif ts.month == 12 and ts.day == 31:
        special_date = "New Year's Eve | New Years Eve | "
    elif ts.month == 1 and ts.day == 1:
        special_date = "New Year's Day | "
    elif ts.month == 12 and ts.day == 26:
        special_date = "Boxing Day | "
    
    return (
        f"game | "
        f"{special_date}"
        f"season:{int(r['season'])} | "
        f"date:{date} | "
        f"day:{day_name} | "
        f"month_day:{month_day} | "
        f"home_team:{home_team} | "
        f"away_team:{away_team} | "
        f"home_team_id:{int(r['home_team_id'])} | "
        f"away_team_id:{int(r['away_team_id'])} | "
        f"home_points:{hp} | "
        f"away_points:{ap} | "
        f"winner:{winner} | "
        f"final_score:{hp}-{ap}"
    )

def row_text_player_box(r):
    """Create embedding text for player_box_scores table"""
    player_name = f"{r['first_name']} {r['last_name']}"
    team_name = r['team_name'] if 'team_name' in r else f"team_{r['team_id']}"
    
    ts = pd.to_datetime(r['game_timestamp'], utc=True)
    date = ts.strftime('%Y-%m-%d')
    
    starter_status = "starter" if r['starter'] == 1 else "bench"
    
    # Calculate total rebounds
    total_reb = int(r['offensive_reb']) + int(r['defensive_reb'])
    
    # Calculate field goal percentages
    fg2_pct = (r['fg2_made'] / r['fg2_attempted'] * 100) if r['fg2_attempted'] > 0 else 0
    fg3_pct = (r['fg3_made'] / r['fg3_attempted'] * 100) if r['fg3_attempted'] > 0 else 0
    
    # Special date handling
    special_date = ""
    if ts.month == 12 and ts.day == 25:
        special_date = "Christmas Day | Christmas | "
    elif ts.month == 12 and ts.day == 24:
        special_date = "Christmas Eve | "
    elif ts.month == 12 and ts.day == 31:
        special_date = "New Year's Eve | New Years Eve | "
    elif ts.month == 1 and ts.day == 1:
        special_date = "New Year's Day | "
    
    return (
        f"player_performance | "
        f"{special_date}"
        f"player:{player_name} | "
        f"team:{team_name} | "
        f"date:{date} | "
        f"starter:{starter_status} | "
        f"points:{int(r['points'])} | "
        f"rebounds:{total_reb} | "
        f"assists:{int(r['assists'])} | "
        f"steals:{int(r['steals'])} | "
        f"blocks:{int(r['blocks'])} | "
        f"turnovers:{int(r['turnovers'])} | "
        f"fg2_made:{int(r['fg2_made'])} | "
        f"fg2_attempted:{int(r['fg2_attempted'])} | "
        f"fg3_made:{int(r['fg3_made'])} | "
        f"fg3_attempted:{int(r['fg3_attempted'])} | "
        f"ft_made:{int(r['ft_made'])} | "
        f"ft_attempted:{int(r['ft_attempted'])} | "
        f"offensive_rebounds:{int(r['offensive_reb'])} | "
        f"defensive_rebounds:{int(r['defensive_reb'])}"
    )

def main():
    print("Starting Embedding Process")
    eng = sa.create_engine(DB_DSN)
    
    with eng.begin() as cx:
        cx.execute(text('ALTER DATABASE nba REFRESH COLLATION VERSION'))
        
        # Add embedding column to game_details
        cx.execute(text("ALTER TABLE IF EXISTS game_details ADD COLUMN IF NOT EXISTS embedding vector(768);"))
        cx.execute(text("CREATE INDEX IF NOT EXISTS idx_game_details_embedding ON game_details USING hnsw (embedding vector_cosine_ops);"))
        
        # Add embedding column to player_box_scores
        cx.execute(text("ALTER TABLE IF EXISTS player_box_scores ADD COLUMN IF NOT EXISTS embedding vector(768);"))
        cx.execute(text("CREATE INDEX IF NOT EXISTS idx_player_box_scores_embedding ON player_box_scores USING hnsw (embedding vector_cosine_ops);"))
        
        # Embed game_details with team names
        print("Embedding game_details...")
        game_query = """
        SELECT gd.*, 
               ht.name as home_team_name, ht.city as home_team_city,
               at.name as away_team_name, at.city as away_team_city
        FROM game_details gd
        JOIN teams ht ON gd.home_team_id = ht.team_id
        JOIN teams at ON gd.away_team_id = at.team_id
        ORDER BY gd.game_timestamp DESC, gd.game_id DESC
        """
        df_games = pd.read_sql(game_query, cx)
        
        for _, r in df_games.iterrows():
            vec = ollama_embed(EMBED_MODEL, row_text_game(r))
            cx.execute(
                text("UPDATE game_details SET embedding = :v WHERE game_id = :gid"), 
                {"v": vec, "gid": int(r['game_id'])}
            )
        print(f"Finished game_details embeddings: {len(df_games)} rows")
        
        # Embed player_box_scores with player and team names
        print("Embedding player_box_scores...")
        player_query = """
        SELECT pbs.*, 
               p.first_name, p.last_name,
               t.name as team_name, t.city as team_city,
               gd.game_timestamp, gd.home_team_id, gd.away_team_id
        FROM player_box_scores pbs
        JOIN players p ON pbs.person_id = p.player_id
        JOIN teams t ON pbs.team_id = t.team_id
        JOIN game_details gd ON pbs.game_id = gd.game_id
        ORDER BY gd.game_timestamp DESC
        """
        df_players = pd.read_sql(player_query, cx)
        
        for _, r in df_players.iterrows():
            vec = ollama_embed(EMBED_MODEL, row_text_player_box(r))
            cx.execute(
                text("UPDATE player_box_scores SET embedding = :v WHERE game_id = :gid AND person_id = :pid"), 
                {"v": vec, "gid": int(r['game_id']), "pid": int(r['person_id'])}
            )
        print(f"Finished player_box_scores embeddings: {len(df_players)} rows")
    
    print(f"Finished All Embeddings")


if __name__ == "__main__":
    main()