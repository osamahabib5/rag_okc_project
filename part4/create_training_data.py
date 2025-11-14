"""
Create training dataset for fine-tuning E5 embedding model.
Generates question-context pairs from NBA game data.
"""

import json
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from backend.config import DB_DSN
from datetime import datetime

def create_training_pairs():
    """Generate question-context pairs from database"""
    eng = sa.create_engine(DB_DSN)
    
    training_pairs = []
    validation_pairs = []
    
    with eng.begin() as cx:
        # Get game data with team information
        game_query = """
        SELECT gd.game_id, gd.season, gd.game_timestamp, 
               gd.home_points, gd.away_points,
               ht.name as home_team, ht.city as home_city,
               at.name as away_team, at.city as away_city,
               wt.name as winning_team
        FROM game_details gd
        JOIN teams ht ON gd.home_team_id = ht.team_id
        JOIN teams at ON gd.away_team_id = at.team_id
        JOIN teams wt ON gd.winning_team_id = wt.team_id
        ORDER BY gd.game_timestamp DESC
        LIMIT 100
        """
        games = pd.read_sql(game_query, cx)
        
        # Get player performance data
        player_query = """
        SELECT pbs.game_id, pbs.points, pbs.offensive_reb + pbs.defensive_reb as rebounds,
               pbs.assists, pbs.steals, pbs.blocks,
               p.first_name, p.last_name,
               t.name as team_name,
               gd.game_timestamp, gd.home_points, gd.away_points,
               ht.name as home_team, at.name as away_team
        FROM player_box_scores pbs
        JOIN players p ON pbs.person_id = p.player_id
        JOIN teams t ON pbs.team_id = t.team_id
        JOIN game_details gd ON pbs.game_id = gd.game_id
        JOIN teams ht ON gd.home_team_id = ht.team_id
        JOIN teams at ON gd.away_team_id = at.team_id
        WHERE pbs.points >= 20
        ORDER BY gd.game_timestamp DESC
        LIMIT 100
        """
        players = pd.read_sql(player_query, cx)
    
    # Generate game result pairs (15 training, 5 validation)
    for idx, game in games.head(20).iterrows():
        date = pd.to_datetime(game['game_timestamp']).strftime('%B %d, %Y')
        home_full = f"{game['home_city']} {game['home_team']}"
        away_full = f"{game['away_city']} {game['away_team']}"
        
        context = f"On {date}, the {home_full} played against the {away_full}. The final score was {home_full} {game['home_points']}, {away_full} {game['away_points']}. The {game['winning_team']} won the game."
        
        # Create multiple question variations
        questions = [
            f"What was the final score between {home_full} and {away_full} on {date}?",
            f"Who won the game between {game['home_team']} and {game['away_team']} on {date}?",
            f"How many points did the {game['home_team']} score against the {game['away_team']} on {date}?",
        ]
        
        pair = {
            "question": questions[idx % 3],
            "context": context,
            "game_id": int(game['game_id'])
        }
        
        if idx < 15:
            training_pairs.append(pair)
        else:
            validation_pairs.append(pair)
    
    # Generate player performance pairs (15 training, 5 validation)
    for idx, player in players.head(20).iterrows():
        date = pd.to_datetime(player['game_timestamp']).strftime('%B %d, %Y')
        player_name = f"{player['first_name']} {player['last_name']}"
        
        context = f"In the game on {date} between {player['home_team']} and {player['away_team']}, {player_name} of the {player['team_name']} scored {player['points']} points, grabbed {player['rebounds']} rebounds, and had {player['assists']} assists."
        
        # Create question variations
        questions = [
            f"How many points did {player_name} score on {date}?",
            f"What were {player_name}'s stats in the game on {date}?",
            f"How did {player_name} perform against {'the ' + player['home_team'] if player['team_name'] != player['home_team'] else 'the ' + player['away_team']} on {date}?",
        ]
        
        pair = {
            "question": questions[idx % 3],
            "context": context,
            "player": player_name
        }
        
        if idx < 15:
            training_pairs.append(pair)
        else:
            validation_pairs.append(pair)
    
    # Generate triple-double pairs (5 training, 2 validation)
    triple_double_query = """
    SELECT pbs.game_id, pbs.points, 
           pbs.offensive_reb + pbs.defensive_reb as rebounds,
           pbs.assists, pbs.steals, pbs.blocks,
           p.first_name, p.last_name,
           t.name as team_name,
           gd.game_timestamp,
           ht.name as home_team, at.name as away_team
    FROM player_box_scores pbs
    JOIN players p ON pbs.person_id = p.player_id
    JOIN teams t ON pbs.team_id = t.team_id
    JOIN game_details gd ON pbs.game_id = gd.game_id
    JOIN teams ht ON gd.home_team_id = ht.team_id
    JOIN teams at ON gd.away_team_id = at.team_id
    WHERE pbs.points >= 10 
      AND (pbs.offensive_reb + pbs.defensive_reb) >= 10 
      AND pbs.assists >= 10
    ORDER BY gd.game_timestamp DESC
    LIMIT 7
    """
    
    with eng.begin() as cx:
        triple_doubles = pd.read_sql(triple_double_query, cx)
    
    for idx, td in triple_doubles.iterrows():
        date = pd.to_datetime(td['game_timestamp']).strftime('%B %d, %Y')
        player_name = f"{td['first_name']} {td['last_name']}"
        
        context = f"{player_name} recorded a triple-double on {date} in the game between {td['home_team']} and {td['away_team']}. He finished with {td['points']} points, {td['rebounds']} rebounds, and {td['assists']} assists for the {td['team_name']}."
        
        question = f"Which player had a triple-double in the {td['home_team']} vs {td['away_team']} game on {date}?"
        
        pair = {
            "question": question,
            "context": context,
            "player": player_name
        }
        
        if idx < 5:
            training_pairs.append(pair)
        else:
            validation_pairs.append(pair)
    
    print(f"Generated {len(training_pairs)} training pairs")
    print(f"Generated {len(validation_pairs)} validation pairs")
    
    return training_pairs, validation_pairs


def save_datasets(training_pairs, validation_pairs):
    """Save datasets to JSON files"""
    with open('training_data.json', 'w') as f:
        json.dump(training_pairs, f, indent=2)
    
    with open('validation_data.json', 'w') as f:
        json.dump(validation_pairs, f, indent=2)
    
    print("Saved training_data.json and validation_data.json")


if __name__ == "__main__":
    training_pairs, validation_pairs = create_training_pairs()
    save_datasets(training_pairs, validation_pairs)
    
    # Print samples
    print("\n=== Sample Training Pair ===")
    print(f"Question: {training_pairs[0]['question']}")
    print(f"Context: {training_pairs[0]['context']}")