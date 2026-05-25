"""Purge forcee des rapports en cache pour forcer la regeneration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.db import get_cursor

with get_cursor() as cur:
    cur.execute("SELECT COUNT(*) AS n FROM rapports_nlp")
    row = cur.fetchone()
    print(f"Rapports en cache avant : {row['n']}")
    cur.execute("TRUNCATE TABLE rapports_nlp")
    cur.execute("TRUNCATE TABLE nltk_analysis")
    print("[OK] rapports_nlp + nltk_analysis purges.")
