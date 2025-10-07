"""
Retrieve and join – Perform semantic retrieval using the pgvector extension to find relevant game summaries,
then join the matched embeddings back to the original structured table rows to provide factual context.
"""

import os
import json
import re
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from datetime import datetime
from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.json"))
TEMPLATE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers_template.json"))


def parse_nba_season_date(question):
    """
    Parse dates considering NBA season logic.
    NBA seasons run from October to June of the following year.
    E.g., "2023 NBA Season" = October 2023 to June 2024
    """
    question_lower = question.lower()
    
    # Look for "YYYY NBA Season" pattern
    season_pattern = r'(\d{4})\s+nba\s+season'
    season_match = re.search(season_pattern, question_lower)
    
    if season_match:
        season_start_year = int(season_match.group(1))
        print(f"Found NBA season: {season_start_year}-{season_start_year + 1}")
        
        # Now look for date pattern like "4/9" or "4-9"
        date_pattern = r'\b(\d{1,2})[/-](\d{1,2})\b'
        date_match = re.search(date_pattern, question)
        
        if date_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            
            # NBA season logic:
            # October (10) - December (12) = season_start_year
            # January (1) - June (6) = season_start_year + 1
            # July (7) - September (9) = off-season, but if mentioned, likely next year
            
            if month >= 10:  # October, November, December
                year = season_start_year
            elif month <= 6:  # January to June
                year = season_start_year + 1
            else:  # July, August, September - typically off-season
                year = season_start_year + 1
            
            date_str = f"{year}-{month:02d}-{day:02d}"
            print(f"Converted date {month}/{day} in {season_start_year} season → {date_str}")
            return [date_str]
    
    return None


def parse_special_dates(question):
    """Parse special dates from question and return list of possible dates"""
    question_lower = question.lower()
    special_dates = []
    
    # First check for NBA season dates
    nba_season_dates = parse_nba_season_date(question)
    if nba_season_dates:
        return nba_season_dates
    
    # Extract year if present
    year_matches = re.findall(r'\b(20\d{2})\b', question)
    years = [int(y) for y in year_matches] if year_matches else []
    
    current_year = datetime.now().year
    
    # Define special date mappings
    special_date_map = {
        'christmas day': (12, 25),
        'christmas': (12, 25),
        'christmas eve': (12, 24),
        "new year's eve": (12, 31),
        'new years eve': (12, 31),
        "new year's day": (1, 1),
        'new years day': (1, 1),
        'boxing day': (12, 26),
        'thanksgiving': None,
        'halloween': (10, 31),
        'valentine': (2, 14),
        "valentine's day": (2, 14),
        'independence day': (7, 4),
        'july fourth': (7, 4),
        '4th of july': (7, 4)
    }
    
    # Check for special dates
    for special_name, date_tuple in special_date_map.items():
        if special_name in question_lower and date_tuple:
            month, day = date_tuple
            
            if years:
                for year in years:
                    special_dates.append(f"{year}-{month:02d}-{day:02d}")
            else:
                for y in [current_year, current_year - 1, current_year - 2]:
                    special_dates.append(f"{y}-{month:02d}-{day:02d}")
    
    # Format: MM/DD/YYYY or M/D/YYYY or M/D/YY
    date_pattern1 = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
    matches1 = re.findall(date_pattern1, question)
    for match in matches1:
        month, day, year = match
        if len(year) == 2:
            year = '20' + year
        try:
            special_dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
        except:
            pass
    
    # Format: "on 4/9" - need to extract year from context
    date_pattern_short = r'\bon\s+(\d{1,2})[/](\d{1,2})\b'
    matches_short = re.findall(date_pattern_short, question_lower)
    for match in matches_short:
        month, day = match
        # Use year from question if available
        if years:
            for year in years:
                try:
                    special_dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
                except:
                    pass
        else:
            # Try current and previous years
            for y in [current_year, current_year - 1, current_year - 2]:
                try:
                    special_dates.append(f"{y}-{int(month):02d}-{int(day):02d}")
                except:
                    pass
    
    # Format: October 27, 2023 or Oct 27, 2023
    month_names = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
        'july': 7, 'jul': 7, 'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10,
        'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    
    date_pattern2 = r'\b(' + '|'.join(month_names.keys()) + r')\.?\s+(\d{1,2}),?\s+(\d{4})\b'
    matches2 = re.findall(date_pattern2, question_lower)
    for match in matches2:
        month_name, day, year = match
        month = month_names[month_name]
        special_dates.append(f"{year}-{month:02d}-{int(day):02d}")
    
    # Format: 1-26-24 or 1/26/24
    date_pattern3 = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2})\b'
    matches3 = re.findall(date_pattern3, question)
    for match in matches3:
        month, day, year = match
        year = '20' + year
        try:
            special_dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
        except:
            pass
    
    return special_dates


def extract_exact_points(question):
    """
    Extract exact point total from question.
    Returns None if not found, or the integer points value.
    """
    # Pattern: "had 40 points" or "40 points" or "scored 40"
    points_patterns = [
        r'had\s+(\d+)\s+points?',
        r'scored\s+(\d+)\s+points?',
        r'(\d+)\s+points?',
        r'with\s+(\d+)\s+points?'
    ]
    
    for pattern in points_patterns:
        match = re.search(pattern, question.lower())
        if match:
            points = int(match.group(1))
            print(f"Extracted exact point total: {points}")
            return points
    
    return None


def extract_player_name(question):
    """Extract player name from question"""
    # Don't extract if the text is about special days
    if any(day in question.lower() for day in ['christmas', 'new year', 'boxing']):
        # Only look for actual player names, not date references
        pass
    
    # Common player name patterns
    player_patterns = [
        r'(LeBron James)',
        r'(Luka Don[cč]i[cć])',
        r'(Victor Wembanyama)',
        r'(Nikola Joki[cć])',
        r'(Shai Gilgeous-Alexander)',
        r'(Stephen Curry)',
        r'(Kevin Durant)',
        r'(Anthony Davis)',
        r'(Giannis Antetokounmpo)',
        r'(Kristaps Porzi[nņ][gģ]i[sš])',
    ]
    
    for pattern in player_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Generic pattern for capitalized names (only if in player context)
    question_lower = question.lower()
    if any(indicator in question_lower for indicator in ['who was', 'who had', 'which player']):
        # Exclude special date names
        if 'christmas' not in question_lower or len(question.split()) > 10:
            name_match = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', question)
            if name_match:
                potential_name = name_match.group(1)
                # Exclude team names and special days
                exclude_terms = ['Golden State', 'Los Angeles', 'San Antonio', 'Oklahoma City', 'New Orleans', 
                               'Sacramento Kings', 'Boston Celtics', 'Dallas Mavericks', 'Christmas Day', 
                               'New Year', 'Boxing Day', 'NBA Season', 'Atlanta Hawks']
                if potential_name not in exclude_terms:
                    return potential_name
    
    return None


def determine_query_type(question):
    """
    Determine if question is about player stats or game details.
    Now checks for cues ANYWHERE in the sentence, not just at the beginning.
    """
    question_lower = question.lower()
    
    # Get first 6 words for analysis (still useful for quick patterns)
    words = question_lower.split()
    first_six = ' '.join(words[:6]) if len(words) >= 6 else question_lower
    
    print(f"Analyzing question: '{question}'")
    print(f"First 6 words: '{first_six}'")
    
    # PRIORITY 1: Check for specific player names ANYWHERE in the question
    player_full_names = [
        'lebron james', 'luka dončić', 'luka doncic', 'victor wembanyama',
        'nikola jokic', 'nikola jokić', 'shai gilgeous-alexander', 'stephen curry',
        'kevin durant', 'anthony davis', 'giannis antetokounmpo',
        'kristaps porzingis', 'kristaps porziņģis'
    ]
    
    for name in player_full_names:
        if name in question_lower:
            print(f"Found player name '{name}' - PLAYER query")
            return 'player'
    
    # PRIORITY 2: Check for player-specific question patterns ANYWHERE
    player_question_patterns = [
        r'how many points did ([a-z]+\s+[a-z]+)',
        r'how many rebounds did ([a-z]+\s+[a-z]+)',
        r'how many assists did ([a-z]+\s+[a-z]+)',
        r'did ([a-z]+\s+[a-z]+) score',
        r'did ([a-z]+\s+[a-z]+) have',
    ]
    
    for pattern in player_question_patterns:
        match = re.search(pattern, question_lower)
        if match:
            name_candidate = match.group(1)
            # Check if it's not a team name
            team_keywords = ['warriors', 'lakers', 'mavericks', 'hawks', 'kings', 'celtics', 
                           'spurs', 'nuggets', 'thunder', 'rockets', 'heat', 'suns']
            if not any(team in name_candidate for team in team_keywords):
                print(f"Found player question pattern with '{name_candidate}' - PLAYER query")
                return 'player'
    
    # PRIORITY 3: Strong player indicators ANYWHERE in the question
    strong_player_indicators = [
        'leading scorer',
        'triple-double',
        'his nba debut',
        'his debut',
        'debut',
        'his performance',
        'player recorded',
        'player had',
        'which player',
        'who was the',
        'who had'
    ]
    
    for indicator in strong_player_indicators:
        if indicator in question_lower:
            print(f"Found player indicator '{indicator}' - PLAYER query")
            return 'player'
    
    # PRIORITY 4: Game/Team patterns at the start (original logic)
    game_starters = [
        'which team won',
        'which team',
        'what team',
        'how many points did the',
        'did the',
        'what was the score',
        'what was the final',
        'final score'
    ]
    
    for starter in game_starters:
        if first_six.startswith(starter):
            print(f"Found game starter '{starter}' - GAME query")
            return 'game'
    
    # PRIORITY 5: Check if asking about team score (with "the" before team name)
    team_score_pattern = r'how many points did the ([a-z\s]+)'
    match = re.search(team_score_pattern, question_lower)
    if match:
        team_candidate = match.group(1).strip()
        team_names = [
            'warriors', 'lakers', 'nuggets', 'kings', 'celtics', 'heat', 'mavericks', 
            'thunder', 'spurs', 'rockets', 'hawks', 'suns', 'clippers'
        ]
        if any(team in team_candidate for team in team_names):
            print(f"Found team score question - GAME query")
            return 'game'
    
    # PRIORITY 6: Strong game indicators ANYWHERE
    strong_game_indicators = [
        'team won',
        'team score',
        'final score',
        'what was the score',
        'game between',
        'victory over',
        'win over',
        'which team'
    ]
    
    for indicator in strong_game_indicators:
        if indicator in question_lower:
            print(f"Found game indicator '{indicator}' - GAME query")
            return 'game'
    
    # PRIORITY 7: Check for score pattern (indicates game query unless player mentioned)
    score_pattern = r'\d{2,3}-\d{2,3}'
    if re.search(score_pattern, question):
        # Score mentioned, but check if it's about a player's performance IN that game
        if extract_player_name(question):
            print(f"Score mentioned but player name found - PLAYER query")
            return 'player'
        else:
            print(f"Score mentioned, no player - GAME query")
            return 'game'
    
    # PRIORITY 8: Default - if we see "against" pattern with player context
    if 'against' in question_lower or 'against the' in question_lower:
        if extract_player_name(question):
            print(f"'against' with player name - PLAYER query")
            return 'player'
        else:
            print(f"'against' without player name - GAME query")
            return 'game'
    
    # Final default
    print(f"No clear pattern found - defaulting to GAME query")
    return 'game'


def extract_all_teams_from_question(question):
    """Extract ALL team names mentioned in the question"""
    question_lower = question.lower()
    
    team_mappings = {
        'warriors': ['warriors', 'golden state', 'gsw'],
        'lakers': ['lakers', 'los angeles lakers', 'la lakers', 'lal'],
        'nuggets': ['nuggets', 'denver', 'den'],
        'kings': ['kings', 'sacramento', 'sac'],
        'celtics': ['celtics', 'boston', 'bos'],
        'heat': ['heat', 'miami', 'mia'],
        'mavericks': ['mavericks', 'dallas', 'mavs', 'dal'],
        'thunder': ['thunder', 'oklahoma city', 'okc'],
        'timberwolves': ['timberwolves', 'minnesota', 'wolves', 'min'],
        'spurs': ['spurs', 'san antonio', 'sas'],
        'rockets': ['rockets', 'houston', 'hou'],
        'hawks': ['hawks', 'atlanta', 'atl'],
        'suns': ['suns', 'phoenix', 'phx'],
        'clippers': ['clippers', 'la clippers', 'lac'],
        'jazz': ['jazz', 'utah', 'uta'],
        'pelicans': ['pelicans', 'new orleans', 'nop'],
        'grizzlies': ['grizzlies', 'memphis', 'mem'],
        'cavaliers': ['cavaliers', 'cleveland', 'cle', 'cavs']
    }
    
    found_teams = []
    
    for canonical_name, aliases in team_mappings.items():
        for alias in aliases:
            if alias in question_lower:
                if canonical_name not in found_teams:
                    found_teams.append(canonical_name)
                break
    
    return found_teams


def extract_team_from_question(question):
    """Extract the PRIMARY team name being asked about in the question"""
    question_lower = question.lower()
    
    team_mappings = {
        'warriors': ['warriors', 'golden state', 'gsw'],
        'lakers': ['lakers', 'los angeles lakers', 'la lakers', 'lal'],
        'nuggets': ['nuggets', 'denver', 'den'],
        'kings': ['kings', 'sacramento', 'sac'],
        'celtics': ['celtics', 'boston', 'bos'],
        'heat': ['heat', 'miami', 'mia'],
        'mavericks': ['mavericks', 'dallas', 'mavs', 'dal'],
        'thunder': ['thunder', 'oklahoma city', 'okc'],
        'timberwolves': ['timberwolves', 'minnesota', 'wolves', 'min'],
        'spurs': ['spurs', 'san antonio', 'sas'],
        'rockets': ['rockets', 'houston', 'hou'],
        'hawks': ['hawks', 'atlanta', 'atl'],
        'suns': ['suns', 'phoenix', 'phx'],
        'clippers': ['clippers', 'la clippers', 'lac'],
        'jazz': ['jazz', 'utah', 'uta'],
        'pelicans': ['pelicans', 'new orleans', 'nop'],
        'grizzlies': ['grizzlies', 'memphis', 'mem']
    }
    
    # Look for possessive form first (e.g., "Mavericks' 148-143 win")
    possessive_pattern = r'([a-z]+)\'s?\s+(?:\d+-\d+\s+)?(?:victory|win)'
    match = re.search(possessive_pattern, question_lower)
    if match:
        team_abbr = match.group(1)
        for canonical_name, aliases in team_mappings.items():
            if team_abbr in aliases or team_abbr in canonical_name:
                return canonical_name
    
    # Try to find team mentioned after "did the" or "did"
    pattern = r'(?:did (?:the\s+)?|score (?:against\s+)?(?:the\s+)?)([a-z\s]+?)(?:\s+score|\s+against|\s+on|$)'
    match = re.search(pattern, question_lower)
    
    if match:
        potential_team = match.group(1).strip()
        for canonical_name, aliases in team_mappings.items():
            if any(alias in potential_team for alias in aliases):
                return canonical_name
    
    # Fallback: search anywhere in question - return first match
    for canonical_name, aliases in team_mappings.items():
        for alias in aliases:
            if alias in question_lower:
                return canonical_name
    
    return None


def extract_opponent_team(question):
    """Extract opponent team from question (e.g., 'against the Mavericks', 'over the Hawks')"""
    question_lower = question.lower()
    
    # Pattern: "against the [Team]" or "vs [Team]" or "over [Team]"
    opponent_patterns = [
        r'(?:victory|win)\s+over\s+(?:the\s+)?([a-z\s]+?)(?:\s+on|\s+\d|,)',
        r'against (?:the\s+)?([a-z\s]+?)(?:\s+on|\s+in|\s+\d|,|$)',
        r'vs\.?\s+(?:the\s+)?([a-z\s]+?)(?:\s+on|\s+in|\s+\d|,|$)',
        r'v\.?\s+(?:the\s+)?([a-z\s]+?)(?:\s+on|\s+in|\s+\d|,|$)',
    ]
    
    for pattern in opponent_patterns:
        match = re.search(pattern, question_lower)
        if match:
            opponent_str = match.group(1).strip()
            # Map to canonical team name
            team_name = extract_team_from_question(opponent_str)
            if team_name:
                return team_name
    
    return None


def extract_score_from_question(question):
    """Extract score from question (e.g., '148-143')"""
    score_pattern = r'(\d{2,3})-(\d{2,3})'
    match = re.search(score_pattern, question)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def find_opponent_team_from_db(cx, opponent_team, dates, player_name=None):
    """
    Find the other team in a game when only one team is mentioned.
    Returns the opponent team and game_id.
    """
    if not opponent_team or not dates:
        return None, None
    
    print(f"Searching for opponent of {opponent_team} on {dates}")
    
    # Query to find the game with the specified team on the specified date
    sql = """
    SELECT gd.game_id, gd.home_team_id, gd.away_team_id, gd.game_timestamp,
           ht.name as home_team_name, ht.abbreviation as home_abbr,
           at.name as away_team_name, at.abbreviation as away_abbr
    FROM game_details gd
    JOIN teams ht ON gd.home_team_id = ht.team_id
    JOIN teams at ON gd.away_team_id = at.team_id
    WHERE DATE(gd.game_timestamp) = :date
    """
    
    for date in dates:
        result = cx.execute(text(sql), {"date": date}).mappings().all()
        
        for row in result:
            home_team = row['home_team_name'].lower()
            away_team = row['away_team_name'].lower()
            
            # Check if opponent_team is in this game
            if opponent_team in home_team or opponent_team in away_team:
                # Determine which team is the OTHER team
                if opponent_team in home_team:
                    other_team = row['away_team_name'].lower()
                    other_team_canonical = extract_team_from_question(other_team)
                    print(f"✓ Found game: {row['home_team_name']} vs {row['away_team_name']}")
                    print(f"  Opponent team: {other_team_canonical}")
                    return other_team_canonical, row['game_id']
                else:
                    other_team = row['home_team_name'].lower()
                    other_team_canonical = extract_team_from_question(other_team)
                    print(f"✓ Found game: {row['home_team_name']} vs {row['away_team_name']}")
                    print(f"  Opponent team: {other_team_canonical}")
                    return other_team_canonical, row['game_id']
    
    print(f"❌ No game found for {opponent_team} on {dates}")
    return None, None


def get_player_stats_directly(cx, player_name, game_id):
    """
    Directly retrieve player stats from database using player name and game_id.
    """
    if not player_name or not game_id:
        return None
    
    print(f"Retrieving stats for {player_name} in game {game_id}")
    
    sql = """
    SELECT pbs.game_id, pbs.person_id, pbs.team_id,
           pbs.starter, pbs.points, pbs.assists, 
           pbs.offensive_reb, pbs.defensive_reb,
           (pbs.offensive_reb + pbs.defensive_reb) as total_rebounds,
           pbs.steals, pbs.blocks, pbs.turnovers,
           pbs.fg2_made, pbs.fg2_attempted,
           pbs.fg3_made, pbs.fg3_attempted,
           pbs.ft_made, pbs.ft_attempted,
           p.first_name, p.last_name,
           t.name as team_name, t.city as team_city, t.abbreviation as team_abbr,
           gd.game_timestamp, gd.home_team_id, gd.away_team_id,
           gd.home_points, gd.away_points, gd.winning_team_id,
           ht.name as home_team_name, ht.abbreviation as home_abbr,
           at.name as away_team_name, at.abbreviation as away_abbr
    FROM player_box_scores pbs
    JOIN players p ON pbs.person_id = p.player_id
    JOIN teams t ON pbs.team_id = t.team_id
    JOIN game_details gd ON pbs.game_id = gd.game_id
    JOIN teams ht ON gd.home_team_id = ht.team_id
    JOIN teams at ON gd.away_team_id = at.team_id
    WHERE pbs.game_id = :game_id
    """
    
    results = cx.execute(text(sql), {"game_id": game_id}).mappings().all()
    
    # Filter by player name
    player_lower = player_name.lower()
    for row in results:
        full_name = f"{row['first_name']} {row['last_name']}".lower()
        if player_lower in full_name or full_name in player_lower:
            print(f"✓ Found player stats: {row['first_name']} {row['last_name']} - {row['points']} pts")
            return [dict(row)]
    
    print(f"❌ Player {player_name} not found in game {game_id}")
    return None


def parse_timestamp(ts):
    """Parse timestamp to datetime object"""
    if isinstance(ts, str):
        return pd.to_datetime(ts, utc=True)
    return ts


def retrieve_games(cx, qvec, k=8):
    """Retrieve relevant games using vector similarity - optimized"""
    sql = """
    SELECT gd.game_id, gd.season, gd.game_timestamp, 
           gd.home_team_id, gd.away_team_id, 
           gd.home_points, gd.away_points, gd.winning_team_id,
           ht.name as home_team_name, ht.city as home_team_city, ht.abbreviation as home_abbr,
           at.name as away_team_name, at.city as away_team_city, at.abbreviation as away_abbr,
           wt.name as winning_team_name, wt.city as winning_team_city, wt.abbreviation as winning_abbr,
           1 - (gd.embedding <=> (:q)::vector) AS score 
    FROM game_details gd
    JOIN teams ht ON gd.home_team_id = ht.team_id
    JOIN teams at ON gd.away_team_id = at.team_id
    JOIN teams wt ON gd.winning_team_id = wt.team_id
    ORDER BY gd.embedding <-> (:q)::vector 
    LIMIT :k
    """
    return cx.execute(text(sql), {"q": qvec, "k": k}).mappings().all()


def retrieve_player_stats(cx, qvec, k=20):
    """Retrieve relevant player performances using vector similarity"""
    sql = """
    SELECT pbs.game_id, pbs.person_id, pbs.team_id,
           pbs.starter, pbs.points, pbs.assists, 
           pbs.offensive_reb, pbs.defensive_reb,
           (pbs.offensive_reb + pbs.defensive_reb) as total_rebounds,
           pbs.steals, pbs.blocks, pbs.turnovers,
           pbs.fg2_made, pbs.fg2_attempted,
           pbs.fg3_made, pbs.fg3_attempted,
           pbs.ft_made, pbs.ft_attempted,
           p.first_name, p.last_name,
           t.name as team_name, t.city as team_city, t.abbreviation as team_abbr,
           gd.game_timestamp, gd.home_team_id, gd.away_team_id,
           gd.home_points, gd.away_points, gd.winning_team_id,
           ht.name as home_team_name, ht.city as home_team_city, ht.abbreviation as home_abbr,
           at.name as away_team_name, at.city as away_team_city, at.abbreviation as away_abbr,
           1 - (pbs.embedding <=> (:q)::vector) AS score
    FROM player_box_scores pbs
    JOIN players p ON pbs.person_id = p.player_id
    JOIN teams t ON pbs.team_id = t.team_id
    JOIN game_details gd ON pbs.game_id = gd.game_id
    JOIN teams ht ON gd.home_team_id = ht.team_id
    JOIN teams at ON gd.away_team_id = at.team_id
    ORDER BY pbs.embedding <-> (:q)::vector
    LIMIT :k
    """
    return cx.execute(text(sql), {"q": qvec, "k": k}).mappings().all()


def filter_by_dates(rows, dates):
    """Filter rows by matching dates"""
    if not dates:
        return rows
    
    filtered = []
    for row in rows:
        ts = parse_timestamp(row['game_timestamp'])
        row_date = ts.strftime('%Y-%m-%d')
        
        if row_date in dates:
            filtered.append(row)
    
    return filtered if filtered else rows


def filter_by_player(rows, player_name):
    """Filter rows by player name"""
    if not player_name:
        return rows
    
    player_lower = player_name.lower()
    filtered = []
    
    for row in rows:
        full_name = f"{row['first_name']} {row['last_name']}".lower()
        if player_lower in full_name or full_name in player_lower:
            filtered.append(row)
    
    return filtered if filtered else rows


def filter_by_matchup(rows, teams):
    """Filter rows by game matchup - BOTH teams must be involved"""
    if not teams or len(teams) < 2:
        return rows
    
    filtered = []
    
    for row in rows:
        home_team = row['home_team_name'].lower()
        away_team = row['away_team_name'].lower()
        
        # Check if all specified teams are in this game
        teams_in_game = set()
        for team in teams:
            if team in home_team or team in away_team:
                teams_in_game.add(team)
        
        # Only include if ALL specified teams are in this game
        if len(teams_in_game) == len(teams):
            filtered.append(row)
    
    return filtered


def filter_by_score(rows, score1, score2):
    """Filter rows by exact game score"""
    if score1 is None or score2 is None:
        return rows
    
    filtered = []
    
    for row in rows:
        home_pts = row['home_points']
        away_pts = row['away_points']
        
        # Check both possible score combinations
        if (home_pts == score1 and away_pts == score2) or (home_pts == score2 and away_pts == score1):
            filtered.append(row)
    
    return filtered


def filter_by_teams(rows, team1, team2=None):
    """Filter rows by team involvement"""
    if not team1:
        return rows
    
    filtered = []
    
    for row in rows:
        home_team = row['home_team_name'].lower()
        away_team = row['away_team_name'].lower()
        
        team1_match = team1 in home_team or team1 in away_team
        team2_match = True if not team2 else (team2 in home_team or team2 in away_team)
        
        if team1_match and team2_match:
            filtered.append(row)
    
    return filtered if filtered else rows


def filter_by_opponent(rows, opponent_team):
    """Filter rows by opponent team"""
    if not opponent_team:
        return rows
    
    filtered = []
    
    for row in rows:
        home_team = row['home_team_name'].lower()
        away_team = row['away_team_name'].lower()
        player_team = row['team_name'].lower() if 'team_name' in row else None
        
        # Determine which team is the opponent
        if player_team:
            # For player stats, opponent is the team they're not on
            if opponent_team in home_team and player_team not in home_team:
                filtered.append(row)
            elif opponent_team in away_team and player_team not in away_team:
                filtered.append(row)
        else:
            # For game stats
            if opponent_team in home_team or opponent_team in away_team:
                filtered.append(row)
    
    return filtered if filtered else rows


def filter_by_exact_points(rows, exact_points):
    """
    Filter players by EXACT point total.
    Returns ONLY players with exactly this many points, or empty list if none.
    """
    if exact_points is None:
        return rows
    
    filtered = [r for r in rows if r['points'] == exact_points]
    
    if filtered:
        print(f"✓ Found {len(filtered)} player(s) with exactly {exact_points} points")
    else:
        print(f"❌ No players found with exactly {exact_points} points")
    
    return filtered


def find_leading_scorer(rows, question):
    """Find the leading scorer from filtered rows for 'leading scorer' questions"""
    if 'leading scorer' not in question.lower():
        return rows
    
    if not rows:
        return rows
    
    # Get all players from the same game
    game_id = rows[0]['game_id']
    same_game_players = [r for r in rows if r['game_id'] == game_id]
    
    # Find max points
    max_points = max(r['points'] for r in same_game_players)
    
    # Get player with max points
    leading_scorer = [r for r in same_game_players if r['points'] == max_points]
    
    print(f"Leading scorer analysis: {len(same_game_players)} players in game {game_id}")
    for r in sorted(same_game_players, key=lambda x: x['points'], reverse=True)[:3]:
        print(f"  {r['first_name']} {r['last_name']} - {r['points']} pts")
    
    return leading_scorer if leading_scorer else rows


def validate_game_context(rows, all_teams, score1, score2):
    """Validate that the retrieved game matches all the criteria from the question"""
    if not rows:
        return False
    
    row = rows[0]
    
    # Check if all mentioned teams are in this game
    if all_teams and len(all_teams) >= 2:
        home_team = row['home_team_name'].lower()
        away_team = row['away_team_name'].lower()
        
        teams_found = 0
        for team in all_teams:
            if team in home_team or team in away_team:
                teams_found += 1
        
        if teams_found != len(all_teams):
            print(f"❌ Team validation failed: Expected {all_teams}, found {teams_found}/{len(all_teams)} teams")
            return False
    
    # Check if score matches (if specified)
    if score1 is not None and score2 is not None:
        home_pts = row['home_points']
        away_pts = row['away_points']
        
        score_match = (home_pts == score1 and away_pts == score2) or (home_pts == score2 and away_pts == score1)
        
        if not score_match:
            print(f"❌ Score validation failed: Expected {score1}-{score2}, found {home_pts}-{away_pts}")
            return False
    
    print(f"✓ Game validation passed")
    return True


def build_game_context(rows, question):
    """Build context string from game rows with explicit home/away labeling"""
    context_lines = []
    asked_team = extract_team_from_question(question)
    
    for idx, r in enumerate(rows):
        if idx >= 5:  # Limit to top 5 for efficiency
            break
            
        ts = parse_timestamp(r['game_timestamp'])
        date = ts.strftime('%Y-%m-%d')
        date_display = ts.strftime('%B %d, %Y')
        
        special_date_label = ""
        if ts.month == 12 and ts.day == 25:
            special_date_label = "Christmas Day | "
        elif ts.month == 12 and ts.day == 24:
            special_date_label = "Christmas Eve | "
        elif ts.month == 12 and ts.day == 31:
            special_date_label = "New Year's Eve | "
        elif ts.month == 1 and ts.day == 1:
            special_date_label = "New Year's Day | "
        
        home_full = f"{r['home_team_city']} {r['home_team_name']}"
        away_full = f"{r['away_team_city']} {r['away_team_name']}"
        winner_full = f"{r['winning_team_city']} {r['winning_team_name']}"
        
        winner_role = "HOME" if r['winning_team_id'] == r['home_team_id'] else "AWAY"
        
        home_team_simple = r['home_team_name'].lower()
        away_team_simple = r['away_team_name'].lower()
        
        asked_team_role = None
        asked_team_points = None
        
        if asked_team:
            if asked_team in home_team_simple:
                asked_team_role = "HOME"
                asked_team_points = r['home_points']
            elif asked_team in away_team_simple:
                asked_team_role = "AWAY"
                asked_team_points = r['away_points']
        
        context_line = (
            f"game_id:{r['game_id']} | {special_date_label}date:{date} | "
            f"HOME:{home_full}({r['home_abbr']}) | HOME_PTS:{r['home_points']} | "
            f"AWAY:{away_full}({r['away_abbr']}) | AWAY_PTS:{r['away_points']} | "
            f"WINNER:{winner_full} | WINNER_WAS:{winner_role} | "
            f"SCORE:{r['home_points']}-{r['away_points']}"
        )
        
        if asked_team_role:
            context_line += f" | ASKED_TEAM_ROLE:{asked_team_role} | ASKED_PTS:{asked_team_points}"
        
        context_lines.append(context_line)
    
    return "\n".join(context_lines)


def build_player_context(rows, question):
    """Build context string from player box score rows"""
    context_lines = []
    
    for idx, r in enumerate(rows):
        if idx >= 3:  # Limit to top 3 for leading scorer questions
            break
            
        ts = parse_timestamp(r['game_timestamp'])
        date = ts.strftime('%Y-%m-%d')
        
        special_date_label = ""
        if ts.month == 12 and ts.day == 25:
            special_date_label = "Christmas | "
        elif ts.month == 12 and ts.day == 31:
            special_date_label = "NYE | "
        
        player = f"{r['first_name']} {r['last_name']}"
        team = f"{r['team_city']} {r['team_name']}"
        home = f"{r['home_team_name']}({r['home_abbr']})"
        away = f"{r['away_team_name']}({r['away_abbr']})"
        
        player_team_role = "HOME" if r['team_id'] == r['home_team_id'] else "AWAY"
        opponent = home if player_team_role == "AWAY" else away
        
        # Check for triple-double
        is_triple_double = sum([
            r['points'] >= 10,
            r['total_rebounds'] >= 10,
            r['assists'] >= 10,
            r['steals'] >= 10,
            r['blocks'] >= 10
        ]) >= 3
        
        context_line = (
            f"game_id:{r['game_id']} | {special_date_label}date:{date} | "
            f"PLAYER:{player} | TEAM:{team}({r['team_abbr']}) | ROLE:{player_team_role} | "
            f"VS:{opponent} | GAME_SCORE:{r['home_points']}-{r['away_points']} | "
            f"PTS:{r['points']} | REB:{r['total_rebounds']} | AST:{r['assists']} | "
            f"STL:{r['steals']} | BLK:{r['blocks']} | TO:{r['turnovers']} | "
            f"FG2:{r['fg2_made']}/{r['fg2_attempted']} | FG3:{r['fg3_made']}/{r['fg3_attempted']} | "
            f"FT:{r['ft_made']}/{r['ft_attempted']} | "
            f"STARTER:{'YES' if r['starter'] == 1 else 'NO'}"
        )
        
        if is_triple_double:
            context_line += " | TRIPLE_DOUBLE:YES"
        
        context_lines.append(context_line)
    
    return "\n".join(context_lines)


def get_template_for_question(question_id):
    """Get the template structure for a specific question"""
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        templates = json.load(f)
    
    for template in templates:
        if template['id'] == question_id:
            return template['result'].copy()
    
    return {}


def extract_structured_answer(llm_response, question_id, query_type, rows, question):
    """Extract structured answer directly from database rows (source of truth)"""
    result = {}
    
    no_info_indicators = [
        'not found', 'no information', 'cannot find', 'not available',
        'no data', 'missing', 'unable to find', 'not in the context',
        'no mention'
    ]
    
    response_lower = llm_response.lower()
    has_no_info = any(indicator in response_lower for indicator in no_info_indicators)
    
    if has_no_info or not rows:
        # Return template with default values
        template = get_template_for_question(question_id)
        if query_type == 'player':
            template['evidence'] = [{"table": "player_box_score", "id": 0}]
        else:
            template['evidence'] = [{"table": "game_details", "id": 0}]
        return template
    
    # USE DATABASE AS SOURCE OF TRUTH - Don't rely on LLM parsing
    if query_type == 'player' and rows:
        top_row = rows[0]  # Already filtered and sorted, so top row is the answer
        
        # Extract player name from database
        result['player_name'] = f"{top_row['first_name']} {top_row['last_name']}"
        
        # Extract points from database
        result['points'] = int(top_row['points'])
        
        # Extract rebounds if relevant
        if 'rebounds' in question.lower() or 'triple' in question.lower():
            result['rebounds'] = int(top_row['total_rebounds'])
        
        # Extract assists if relevant
        if 'assists' in question.lower() or 'triple' in question.lower():
            result['assists'] = int(top_row['assists'])
        
        # Build evidence
        result['evidence'] = [{"table": "player_box_score", "id": int(top_row['game_id'])}]
    
    # For game queries
    elif query_type == 'game' and rows:
        top_row = rows[0]
        
        # Extract points - try LLM first as fallback
        points_match = re.search(r'(\d+)\s*points?', llm_response, re.IGNORECASE)
        if points_match:
            result['points'] = int(points_match.group(1))
        
        # Extract winner
        winner_match = re.search(r'(?:winner|won)[:\s]+([A-Za-z\s]+?)(?:\s*\(|\s*score|\s*with|\.|$)', llm_response, re.IGNORECASE)
        if winner_match:
            result['winner'] = winner_match.group(1).strip()
        
        # Extract score
        score_match = re.search(r'(\d+)-(\d+)', llm_response)
        if score_match:
            result['score'] = f"{score_match.group(1)}-{score_match.group(2)}"
        
        # Build evidence
        result['evidence'] = [{"table": "game_details", "id": int(top_row['game_id'])}]
    
    else:
        # Fallback - return template
        template = get_template_for_question(question_id)
        if query_type == 'player':
            template['evidence'] = [{"table": "player_box_score", "id": 0}]
        else:
            template['evidence'] = [{"table": "game_details", "id": 0}]
        return template
    
    return result


def answer_question(question, question_id, cx):
    """Answer a question using RAG - optimized"""
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
    
    if special_dates:
        print(f"Dates: {special_dates}")
    if exact_points:
        print(f"Exact points required: {exact_points}")
    if all_teams:
        print(f"All teams mentioned: {all_teams}")
    if player_name:
        print(f"Player: {player_name}")
    if asked_team:
        print(f"Primary team: {asked_team}")
    if opponent_team:
        print(f"Opponent: {opponent_team}")
    if score1 and score2:
        print(f"Score: {score1}-{score2}")
    
    # SPECIAL LOGIC: If player name + date + one team is given, find the other team
    if query_type == 'player' and player_name and special_dates and len(all_teams) == 1 and not exact_points:
        print(f"🔍 Special case: Player + Date + One Team - Finding opponent...")
        mentioned_team = all_teams[0]
        other_team, game_id = find_opponent_team_from_db(cx, mentioned_team, special_dates, player_name)
        
        if other_team and game_id:
            # Ensure both teams are different
            if other_team == mentioned_team:
                print(f"❌ Error: Both teams are the same ({mentioned_team})")
                template = get_template_for_question(question_id)
                template['evidence'] = [{"table": "player_box_score", "id": 0}]
                return template, [], "Invalid game: both teams are the same."
            
            # Get player stats directly from database
            rows = get_player_stats_directly(cx, player_name, game_id)
            
            if rows:
                print(f"✓ Retrieved player stats directly from database")
                ctx = build_player_context(rows, question)
                
                prompt = f"""Answer using ONLY the context below.

Context:
{ctx}

Question: {question}

Instructions:
- Extract: PLAYER name, PTS (points)
- Use EXACT numbers from the context
- Format: "[Player Name] | Points: [X] | game_id: [ID]"

Answer:"""
                
                llm_response = ollama_generate(LLM_MODEL, prompt)
                result = extract_structured_answer(llm_response, question_id, query_type, rows, question)
                return result, rows, llm_response
            else:
                print(f"❌ Player stats not found")
                template = get_template_for_question(question_id)
                template['evidence'] = [{"table": "player_box_score", "id": 0}]
                return template, [], "Player stats not found in the database."
        else:
            print(f"❌ Could not find the game")
            template = get_template_for_question(question_id)
            template['evidence'] = [{"table": "player_box_score", "id": 0}]
            return template, [], "Game not found in the database."
    
    # REGULAR LOGIC: Use embeddings for retrieval
    qvec = ollama_embed(EMBED_MODEL, question)
    
    if query_type == 'player':
        rows = retrieve_player_stats(cx, qvec, k=20)
        
        # Apply filters
        if special_dates:
            rows = filter_by_dates(rows, special_dates)
            print(f"After date filter: {len(rows)} rows")
        
        # Filter by matchup (both teams must be in the game)
        if all_teams and len(all_teams) >= 2:
            rows = filter_by_matchup(rows, all_teams)
            print(f"After matchup filter ({all_teams}): {len(rows)} rows")
        
        # Filter by score if specified
        if score1 and score2:
            rows = filter_by_score(rows, score1, score2)
            print(f"After score filter ({score1}-{score2}): {len(rows)} rows")
        
        # Validate the game matches all criteria
        if all_teams and len(all_teams) >= 2:
            if not validate_game_context(rows, all_teams, score1, score2):
                print("❌ Game validation failed - returning empty result")
                template = get_template_for_question(question_id)
                template['evidence'] = [{"table": "player_box_score", "id": 0}]
                return template, [], "The specified game was not found in the database."
        
        if player_name:
            rows = filter_by_player(rows, player_name)
            print(f"After player filter: {len(rows)} rows")
        
        # CRITICAL: Filter by exact points if specified (for questions like "40 points on 4/9")
        if exact_points is not None:
            rows = filter_by_exact_points(rows, exact_points)
            print(f"After exact points filter ({exact_points}): {len(rows)} rows")
            
            # If no one scored exactly that many points, return empty
            if not rows:
                print(f"❌ No player scored exactly {exact_points} points on the specified date")
                template = get_template_for_question(question_id)
                template['evidence'] = [{"table": "player_box_score", "id": 0}]
                return template, [], f"No player scored exactly {exact_points} points on the specified date."
        
        if opponent_team and not all_teams:
            rows = filter_by_opponent(rows, opponent_team)
            print(f"After opponent filter: {len(rows)} rows")
        
        if asked_team and not all_teams:
            rows = filter_by_teams(rows, asked_team)
            print(f"After team filter: {len(rows)} rows")
        
        # Special handling for "leading scorer" questions (only if no exact points specified)
        if exact_points is None:
            rows = find_leading_scorer(rows, question)
        
        if not rows:
            print("No relevant player data found")
            template = get_template_for_question(question_id)
            template['evidence'] = [{"table": "player_box_score", "id": 0}]
            return template, [], "Information not available in the database."
        
        # Print best match for debugging with game info
        if rows:
            ts = parse_timestamp(rows[0]['game_timestamp'])
            date = ts.strftime('%Y-%m-%d')
            print(f"Best match: {rows[0]['first_name']} {rows[0]['last_name']} - {rows[0]['points']} pts on {date}")
            print(f"  Game: {rows[0]['home_team_name']} vs {rows[0]['away_team_name']}")
        
        ctx = build_player_context(rows, question)
        
        prompt = f"""Answer using ONLY the context below. Use the FIRST entry as the answer.

Context:
{ctx}

Question: {question}

Instructions:
- The FIRST entry in the context is the correct answer
- Extract: PLAYER name, PTS (points), REB (rebounds), AST (assists)
- Use EXACT numbers from the context
- Format: "[Player Name] | Points: [X] | Rebounds: [Y] | Assists: [Z] | game_id: [ID]"

Answer:"""
    else:
        rows = retrieve_games(cx, qvec, k=8)
        
        if special_dates:
            rows = filter_by_dates(rows, special_dates)
        if asked_team and opponent_team:
            rows = filter_by_teams(rows, asked_team, opponent_team)
        elif asked_team or opponent_team:
            rows = filter_by_teams(rows, asked_team or opponent_team)
        
        if not rows:
            print("No relevant game data found")
            template = get_template_for_question(question_id)
            template['evidence'] = [{"table": "game_details", "id": 0}]
            return template, [], "Information not available in the database."
        
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
- Format: "[Team] scored [X] points. Winner: [Team] ([HOME/AWAY]). Score: [X-Y]. game_id: [ID]"

Answer:"""
    
    llm_response = ollama_generate(LLM_MODEL, prompt)
    
    # Extract answer directly from database rows (source of truth)
    result = extract_structured_answer(llm_response, question_id, query_type, rows, question)
    
    return result, rows, llm_response


def main():
    print("Starting RAG Question Answering")
    eng = sa.create_engine(DB_DSN)
    
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        answers_template = json.load(f)
    
    results = []
    
    with eng.begin() as cx:
        for q in questions:
            print(f"\n{'='*80}")
            print(f"Q{q['id']}: {q['question']}")
            print(f"{'='*80}")
            
            result, rows, llm_response = answer_question(q['question'], q['id'], cx)
            
            print(f"Response: {llm_response}")
            print(f"Result: {result}")
            
            # Use the result directly (already in correct format)
            results.append({"id": q['id'], "result": result})
    
    # Write results
    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Finished! Answers written to {ANSWERS_PATH}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()