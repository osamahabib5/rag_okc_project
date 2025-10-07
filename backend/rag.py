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


def parse_special_dates(question):
    """Parse special dates from question and return list of possible dates"""
    question_lower = question.lower()
    special_dates = []
    
    # Extract year if present
    year_match = re.search(r'\b(20\d{2})\b', question)
    
    # Extract season year (e.g., "2023 NBA Season" -> 2023)
    season_match = re.search(r'(\d{4})\s+(?:nba\s+)?season', question_lower)
    
    if season_match:
        year = int(season_match.group(1))
    elif year_match:
        year = int(year_match.group(1))
    else:
        year = datetime.now().year
    
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
            special_dates.append(f"{year}-{month:02d}-{day:02d}")
    
    # Format: M/D pattern (e.g., "4/9" -> April 9)
    date_pattern_short = r'\b(\d{1,2})/(\d{1,2})\b'
    matches_short = re.findall(date_pattern_short, question)
    for match in matches_short:
        month, day = match
        try:
            special_dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
        except:
            pass
    
    # Format: MM/DD/YYYY or M/D/YYYY
    date_pattern1 = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
    matches1 = re.findall(date_pattern1, question)
    for match in matches1:
        month, day, year_str = match
        if len(year_str) == 2:
            year_str = '20' + year_str
        try:
            special_dates.append(f"{year_str}-{int(month):02d}-{int(day):02d}")
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
        month_name, day, year_str = match
        month = month_names[month_name]
        special_dates.append(f"{year_str}-{month:02d}-{int(day):02d}")
    
    # Format: 1-26-24 or 1/26/24
    date_pattern3 = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2})\b'
    matches3 = re.findall(date_pattern3, question)
    for match in matches3:
        month, day, year_str = match
        year_str = '20' + year_str
        try:
            special_dates.append(f"{year_str}-{int(month):02d}-{int(day):02d}")
        except:
            pass
    
    return special_dates


def extract_player_name(question):
    """Extract player name from question"""
    # Don't extract if the text is about special days
    if any(day in question.lower() for day in ['christmas', 'new year', 'boxing']):
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
                               'New Year', 'Boxing Day', 'NBA Season']
                if potential_name not in exclude_terms:
                    return potential_name
    
    return None


def determine_query_type(question):
    """Determine if question is about player stats or game details based on first words"""
    question_lower = question.lower()
    
    # Get first 6 words for analysis
    words = question_lower.split()
    first_six = ' '.join(words[:6]) if len(words) >= 6 else question_lower
    
    print(f"First 6 words: '{first_six}'")
    
    # CRITICAL: Check first words for definitive patterns
    
    # Game/Team patterns at the start
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
            # Exception: if followed by a known player name, it's a player query
            player_names_lower = ['lebron', 'luka', 'victor', 'nikola', 'shai', 'stephen', 'kevin', 'giannis']
            if not any(name in question_lower for name in player_names_lower):
                return 'game'
    
    # Player patterns at the start
    player_starters = [
        'who was the leading',
        'who was',
        'who had',
        'which player',
        'how many points did lebron',
        'how many points did luka',
        'how many points did victor',
        'how many points did nikola',
        'how many points did shai'
    ]
    
    for starter in player_starters:
        if first_six.startswith(starter):
            return 'player'
    
    # Check if question starts with "How many points did [Name]"
    how_many_pattern = r'^how many points did ([a-z]+)'
    match = re.match(how_many_pattern, question_lower)
    if match:
        first_word_after = match.group(1)
        
        # Common first names of players
        player_first_names = ['lebron', 'luka', 'victor', 'nikola', 'shai', 'stephen', 'kevin', 'giannis', 'anthony']
        
        # Team names/cities that could follow "the"
        team_identifiers = ['warriors', 'lakers', 'nuggets', 'kings', 'celtics', 'heat', 'mavericks', 
                           'thunder', 'spurs', 'rockets', 'hawks', 'suns', 'clippers']
        
        if first_word_after in player_first_names:
            return 'player'
        elif first_word_after == 'the':
            return 'game'
        elif first_word_after in team_identifiers:
            return 'game'
    
    # Strong player indicators anywhere in question
    strong_player_indicators = [
        'leading scorer',
        'triple-double',
        'his nba debut',
        'his debut',
        'his performance',
        'player recorded',
        'player had',
        'which player had'
    ]
    
    for indicator in strong_player_indicators:
        if indicator in question_lower:
            return 'player'
    
    # Check for specific player names (full names)
    player_full_names = [
        'lebron james', 'luka dončić', 'luka doncic', 'victor wembanyama',
        'nikola jokic', 'shai gilgeous-alexander', 'stephen curry',
        'kevin durant', 'anthony davis', 'giannis antetokounmpo'
    ]
    
    for name in player_full_names:
        if name in question_lower:
            return 'player'
    
    # Strong game indicators
    strong_game_indicators = [
        'team won',
        'team score',
        'final score',
        'what was the score',
        'game between',
        'victory over',
        'win over'
    ]
    
    for indicator in strong_game_indicators:
        if indicator in question_lower:
            return 'game'
    
    # Check for pattern "against the [Team]" - this is typically about teams unless player name is present
    if 'against the' in question_lower or 'against' in question_lower:
        if extract_player_name(question):
            return 'player'
        else:
            return 'game'
    
    # Default to game when uncertain
    return 'game'


def extract_all_teams_from_question(question):
    """Extract all teams mentioned in the question (for player queries about specific matchups)"""
    question_lower = question.lower()
    teams_found = []
    
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
    
    # Search for all team mentions
    for canonical_name, aliases in team_mappings.items():
        for alias in aliases:
            if alias in question_lower:
                if canonical_name not in teams_found:
                    teams_found.append(canonical_name)
                break
    
    return teams_found


def extract_team_from_question(question):
    """Extract the primary team name being asked about in the question"""
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
    
    # For player queries with possessive (e.g., "OKC's victory"), extract the possessive team
    possessive_pattern = r"([a-z]+)'s\s+(?:victory|win|game)"
    possessive_match = re.search(possessive_pattern, question_lower)
    if possessive_match:
        team_mention = possessive_match.group(1)
        for canonical_name, aliases in team_mappings.items():
            if any(alias == team_mention or alias in team_mention for alias in aliases):
                return canonical_name
    
    # Try to find team mentioned after "did the" or "did"
    pattern = r'(?:did (?:the\s+)?|score (?:against\s+)?(?:the\s+)?)([a-z\s]+?)(?:\s+score|\s+against|\s+on|$)'
    match = re.search(pattern, question_lower)
    
    if match:
        potential_team = match.group(1).strip()
        for canonical_name, aliases in team_mappings.items():
            if any(alias in potential_team for alias in aliases):
                return canonical_name
    
    # Fallback: search anywhere in question (return first match)
    for canonical_name, aliases in team_mappings.items():
        for alias in aliases:
            if alias in question_lower:
                return canonical_name
    
    return None


def extract_opponent_team(question):
    """Extract opponent team from question (e.g., 'against the Mavericks', 'over SAC')"""
    question_lower = question.lower()
    
    # Pattern: "over [Team]" or "against [Team]" or "vs [Team]"
    opponent_patterns = [
        r'over\s+(?:the\s+)?([a-z]+?)(?:\s+on|\s+in|\s+\d|,|$)',
        r'against (?:the\s+)?([a-z\s]+?)(?:\s+on|\s+in|\s+\d|$)',
        r'vs\.?\s+(?:the\s+)?([a-z\s]+?)(?:\s+on|\s+in|\s+\d|$)',
        r'v\.?\s+(?:the\s+)?([a-z\s]+?)(?:\s+on|\s+in|\s+\d|$)',
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


def extract_points_from_question(question):
    """Extract specific points value mentioned in question (e.g., '40 points')"""
    question_lower = question.lower()
    
    # Pattern: "had X points" or "X points on"
    points_patterns = [
        r'had\s+(\d+)\s+points?',
        r'(\d+)\s+points?\s+on',
        r'scored\s+(\d+)\s+points?'
    ]
    
    for pattern in points_patterns:
        match = re.search(pattern, question_lower)
        if match:
            return int(match.group(1))
    
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


def filter_by_points(rows, points):
    """Filter rows by specific points scored"""
    if not points:
        return rows
    
    filtered = []
    for row in rows:
        if row['points'] == points:
            filtered.append(row)
    
    return filtered if filtered else rows


def filter_by_matchup(rows, teams_list):
    """Filter rows by specific matchup (both teams must be involved in the game)"""
    if not teams_list or len(teams_list) < 2:
        return rows
    
    filtered = []
    
    for row in rows:
        home_team = row['home_team_name'].lower()
        away_team = row['away_team_name'].lower()
        
        # Check if both teams from the list are in this game
        teams_in_game = []
        for team in teams_list:
            if team in home_team or team in away_team:
                teams_in_game.append(team)
        
        # If we have at least 2 teams from the list in this game, it's a match
        if len(teams_in_game) >= 2:
            filtered.append(row)
    
    return filtered if filtered else rows


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
        
        if player_team:
            if opponent_team in home_team and player_team not in home_team:
                filtered.append(row)
            elif opponent_team in away_team and player_team not in away_team:
                filtered.append(row)
        else:
            if opponent_team in home_team or opponent_team in away_team:
                filtered.append(row)
    
    return filtered if filtered else rows


def find_leading_scorer(rows, question):
    """Find the leading scorer from filtered rows for 'leading scorer' questions"""
    if 'leading scorer' not in question.lower():
        return rows
    
    if not rows:
        return rows
    
    game_id = rows[0]['game_id']
    same_game_players = [r for r in rows if r['game_id'] == game_id]
    
    max_points = max(r['points'] for r in same_game_players)
    leading_scorer = [r for r in same_game_players if r['points'] == max_points]
    
    print(f"Leading scorer analysis: {len(same_game_players)} players in game {game_id}")
    for r in sorted(same_game_players, key=lambda x: x['points'], reverse=True)[:3]:
        print(f"  {r['first_name']} {r['last_name']} - {r['points']} pts")
    
    return leading_scorer if leading_scorer else rows


def build_game_context(rows, question):
    """Build context string from game rows with explicit home/away labeling"""
    context_lines = []
    asked_team = extract_team_from_question(question)
    
    for idx, r in enumerate(rows):
        if idx >= 5:
            break
            
        ts = parse_timestamp(r['game_timestamp'])
        date = ts.strftime('%Y-%m-%d')
        
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
        if idx >= 5:
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
            f"VS:{opponent} | MATCHUP:{home} vs {away} | "
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
        'no player', 'no game'
    ]
    
    response_lower = llm_response.lower()
    has_no_info = any(indicator in response_lower for indicator in no_info_indicators)
    
    # If no data found or LLM indicates no info, return template defaults
    if has_no_info or not rows:
        print("No data found - returning template defaults")
        template = get_template_for_question(question_id)
        return template
    
    # USE DATABASE AS SOURCE OF TRUTH
    if query_type == 'player' and rows:
        top_row = rows[0]
        
        result['player_name'] = f"{top_row['first_name']} {top_row['last_name']}"
        result['points'] = int(top_row['points'])
        
        if 'rebounds' in question.lower() or 'triple' in question.lower():
            result['rebounds'] = int(top_row['total_rebounds'])
        
        if 'assists' in question.lower() or 'triple' in question.lower():
            result['assists'] = int(top_row['assists'])
        
        result['evidence'] = [{"table": "player_box_score", "id": int(top_row['game_id'])}]
    
    elif query_type == 'game' and rows:
        top_row = rows[0]
        
        points_match = re.search(r'(\d+)\s*points?', llm_response, re.IGNORECASE)
        if points_match:
            result['points'] = int(points_match.group(1))
        
        winner_match = re.search(r'(?:winner|won)[:\s]+([A-Za-z\s]+?)(?:\s*\(|\s*score|\s*with|\.|$)', llm_response, re.IGNORECASE)
        if winner_match:
            result['winner'] = winner_match.group(1).strip()
        
        score_match = re.search(r'(\d+)-(\d+)', llm_response)
        if score_match:
            result['score'] = f"{score_match.group(1)}-{score_match.group(2)}"
        
        result['evidence'] = [{"table": "game_details", "id": int(top_row['game_id'])}]
    
    else:
        print("Unexpected query type - returning template defaults")
        template = get_template_for_question(question_id)
        return template
    
    return result


def answer_question(question, question_id, cx):
    """Answer a question using RAG"""
    query_type = determine_query_type(question)
    print(f"Query type: {query_type}")
    
    # Extract contextual information
    special_dates = parse_special_dates(question)
    player_name = extract_player_name(question)
    all_teams = extract_all_teams_from_question(question)
    asked_team = extract_team_from_question(question)
    opponent_team = extract_opponent_team(question)
    specific_points = extract_points_from_question(question)
    
    if special_dates:
        print(f"Dates: {special_dates}")
    if player_name:
        print(f"Player: {player_name}")
    if all_teams:
        print(f"All teams mentioned: {all_teams}")
    if asked_team:
        print(f"Primary team: {asked_team}")
    if opponent_team:
        print(f"Opponent: {opponent_team}")
    if specific_points:
        print(f"Specific points: {specific_points}")
    
    qvec = ollama_embed(EMBED_MODEL, question)
    
    if query_type == 'player':
        rows = retrieve_player_stats(cx, qvec, k=20)
        
        # Apply filters in order of specificity
        if special_dates:
            rows = filter_by_dates(rows, special_dates)
            print(f"After date filter: {len(rows)} rows")
        
        # If multiple teams mentioned (e.g., "OKC's victory over SAC"), filter by matchup
        if len(all_teams) >= 2:
            rows = filter_by_matchup(rows, all_teams)
            print(f"After matchup filter ({all_teams}): {len(rows)} rows")
        elif asked_team and opponent_team:
            # Use both team filters
            rows = filter_by_teams(rows, asked_team, opponent_team)
            print(f"After team filter ({asked_team} vs {opponent_team}): {len(rows)} rows")
        elif asked_team:
            rows = filter_by_teams(rows, asked_team)
            print(f"After team filter ({asked_team}): {len(rows)} rows")
        elif opponent_team:
            rows = filter_by_opponent(rows, opponent_team)
            print(f"After opponent filter ({opponent_team}): {len(rows)} rows")
        
        if specific_points:
            rows = filter_by_points(rows, specific_points)
            print(f"After points filter ({specific_points}): {len(rows)} rows")
        
        if player_name:
            rows = filter_by_player(rows, player_name)
            print(f"After player filter ({player_name}): {len(rows)} rows")
        
        # Special handling for "leading scorer" questions
        rows = find_leading_scorer(rows, question)
        
        if not rows:
            print("No relevant player data found after filtering")
            template = get_template_for_question(question_id)
            return template, [], "No data found matching the criteria."
        
        if rows:
            print(f"Best match: {rows[0]['first_name']} {rows[0]['last_name']} - {rows[0]['points']} pts on {parse_timestamp(rows[0]['game_timestamp']).strftime('%Y-%m-%d')}")
            print(f"  Game: {rows[0]['home_team_name']} vs {rows[0]['away_team_name']}")
        
        ctx = build_player_context(rows, question)
        
        prompt = f"""Answer using ONLY the context below. Use the FIRST entry as the answer.

Context:
{ctx}

Question: {question}

Instructions:
- The FIRST entry in the context is the correct answer
- Look at the MATCHUP field to see which teams played
- Extract: PLAYER name, PTS (points), REB (rebounds), AST (assists)
- Use EXACT numbers from the context
- If no matching data exists, say "No data found"
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
            print("No relevant game data found after filtering")
            template = get_template_for_question(question_id)
            return template, [], "No data found matching the criteria."
        
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
- If no matching data exists, say "No data found"
- Format: "[Team] scored [X] points. Winner: [Team] ([HOME/AWAY]). Score: [X-Y]. game_id: [ID]"

Answer:"""
    
    llm_response = ollama_generate(LLM_MODEL, prompt)
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
    
    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Finished! Answers written to {ANSWERS_PATH}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()