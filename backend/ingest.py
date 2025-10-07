"""
Load data – Ingest CSVs related to NBA game information from the 2023-24 and 2024-25 seasons into PostgresSQL tables.
"""

import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from pathlib import Path
from backend.config import DB_DSN

TABLES = ["game_details", "player_box_scores", "players", "teams"]
DATA_DIR = Path(__file__).resolve().parent / "data"

def handle_missing_values(df, table_name):
    """Handle missing values based on column type"""
    for col in df.columns:
        if df[col].isnull().any():
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's discrete (integer) or continuous (float)
                if pd.api.types.is_integer_dtype(df[col]):
                    # Use mode for discrete numeric
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
                else:
                    # Use median for continuous numeric
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                # Use mode for non-numeric (categorical)
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('', inplace=True)
    return df

def main():
    print('Starting Database Ingestion')
    eng = sa.create_engine(DB_DSN)
    with eng.begin() as cx:
        # Ensure pgvector extension is available for the `vector` type used to store embeddings
        cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for t in TABLES:
            path = os.path.join(DATA_DIR, f"{t}.csv")
            df = pd.read_csv(path)
            
            # Handle missing values
            df = handle_missing_values(df, t)
            
            df.to_sql(t, cx, if_exists="replace", index=False, method="multi", chunksize=5000)
            print(f"Loaded {len(df)} rows into {t}")
    print('Finished Database Ingestion')


if __name__ == "__main__":
    main()