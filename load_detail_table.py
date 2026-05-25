"""
load_detail_table.py - Chargement du CSV source dans ventes_detail
====================================================================

Role
----
Charge le fichier `data/train.csv` (Superstore Sales, environ 9 800
lignes) dans la table PostgreSQL `ventes_detail`. Cette table sert de
source unique de verite pour tous les KPIs filtres : toute combinaison
year + region + category passe par une agregation sur ventes_detail
dans `src/db.py`.

Pourquoi cette table existe
---------------------------
Les tables kpi_* sont pre-agregees par Spark, donc figees. Pour
permettre au dashboard et au PDF d'appliquer N'IMPORTE QUEL filtre,
on stocke aussi le grain (une ligne par article de commande). Toutes
les queries filtres dans db.py partent de ventes_detail.

Schema
------
    id, order_id, order_date, annee, trimestre, mois, customer_id,
    segment, region, category, sub_category, product_name, sales

Avec 6 index pour accelerer les filtres usuels :
    idx_vd_annee, idx_vd_region, idx_vd_category, idx_vd_sub_category,
    idx_vd_segment, idx_vd_composite(annee, region, category).

Performance
-----------
Insertion en BATCH via psycopg2.extras.execute_values (gain x10 par
rapport a un INSERT par ligne sur 9800 lignes).

Mode d'emploi
-------------
    python load_detail_table.py
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from psycopg2.extras import execute_values

try:
    from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
except ImportError:
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "nlp_reporting"
    DB_USER = "postgres"
    DB_PASSWORD = "admin123"

CSV_PATH = PROJECT_ROOT / "data" / "train.csv"
BATCH_SIZE = 1000


def create_table(conn):
    """Recree la table ventes_detail et ses index (DROP IF EXISTS prealable)."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS ventes_detail;")
        cur.execute(
            """
            CREATE TABLE ventes_detail (
                id SERIAL PRIMARY KEY,
                order_id VARCHAR(50),
                order_date DATE,
                annee INTEGER,
                trimestre INTEGER,
                mois INTEGER,
                customer_id VARCHAR(50),
                segment VARCHAR(50),
                region VARCHAR(50),
                category VARCHAR(50),
                sub_category VARCHAR(50),
                product_name VARCHAR(500),
                sales NUMERIC(12,2)
            );
            """
        )
        cur.execute("CREATE INDEX idx_vd_annee ON ventes_detail(annee);")
        cur.execute("CREATE INDEX idx_vd_region ON ventes_detail(region);")
        cur.execute("CREATE INDEX idx_vd_category ON ventes_detail(category);")
        cur.execute("CREATE INDEX idx_vd_sub_category ON ventes_detail(sub_category);")
        cur.execute("CREATE INDEX idx_vd_segment ON ventes_detail(segment);")
        cur.execute("CREATE INDEX idx_vd_composite ON ventes_detail(annee, region, category);")
        conn.commit()
        print("[OK] Table ventes_detail creee avec index")


def _parse_row(row, line_num):
    """Parse une ligne CSV en tuple SQL. Renvoie None si la ligne est invalide."""
    try:
        dt = datetime.strptime(row["Order Date"], "%d/%m/%Y")
        annee = dt.year
        mois = dt.month
        trimestre = (mois - 1) // 3 + 1
        sales = float(row["Sales"])
        return (
            row["Order ID"], dt.date(), annee, trimestre, mois,
            row["Customer ID"], row["Segment"], row["Region"],
            row["Category"], row["Sub-Category"],
            row["Product Name"], sales,
        )
    except Exception as e:
        print(f"[WARN] Erreur ligne {line_num}: {e}")
        return None


def load_csv(conn):
    """Insertion en batch via execute_values (10x plus rapide que INSERT ligne par ligne)."""
    if not CSV_PATH.exists():
        print(f"[KO] CSV introuvable: {CSV_PATH}")
        return 0

    count = 0
    errors = 0
    batch = []
    line_num = 0

    insert_sql = """
        INSERT INTO ventes_detail
        (order_id, order_date, annee, trimestre, mois,
         customer_id, segment, region, category, sub_category,
         product_name, sales)
        VALUES %s
    """

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        with conn.cursor() as cur:
            for row in reader:
                line_num += 1
                parsed = _parse_row(row, line_num)
                if parsed is None:
                    errors += 1
                    continue
                batch.append(parsed)
                if len(batch) >= BATCH_SIZE:
                    execute_values(cur, insert_sql, batch, page_size=BATCH_SIZE)
                    count += len(batch)
                    print(f"   Batch insere : {count} lignes cumulees")
                    batch = []
            if batch:
                execute_values(cur, insert_sql, batch, page_size=BATCH_SIZE)
                count += len(batch)
            conn.commit()

    print(f"[OK] {count} lignes inserees dans ventes_detail")
    if errors:
        print(f"[WARN] {errors} ligne(s) ignoree(s) (erreurs de parsing)")
    return count


def verify(conn):
    """Verification basique : total de lignes, valeurs distinctes, test de filtre."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ventes_detail;")
        total = cur.fetchone()[0]

        cur.execute("SELECT DISTINCT annee FROM ventes_detail ORDER BY annee;")
        years = [r[0] for r in cur.fetchall()]

        cur.execute("SELECT DISTINCT region FROM ventes_detail ORDER BY region;")
        regions = [r[0] for r in cur.fetchall()]

        cur.execute("SELECT DISTINCT category FROM ventes_detail ORDER BY category;")
        cats = [r[0] for r in cur.fetchall()]

        cur.execute("SELECT DISTINCT sub_category FROM ventes_detail ORDER BY sub_category;")
        subs = [r[0] for r in cur.fetchall()]

        cur.execute("SELECT DISTINCT segment FROM ventes_detail ORDER BY segment;")
        segs = [r[0] for r in cur.fetchall()]

        cur.execute(
            """
            SELECT SUM(sales) as ca, COUNT(*) as nb
            FROM ventes_detail
            WHERE annee = 2018 AND region = 'West' AND category = 'Furniture'
            """
        )
        test = cur.fetchone()

        print("\n  Verification:")
        print(f"   Total: {total} lignes")
        print(f"   Annees: {years}")
        print(f"   Regions: {regions}")
        print(f"   Categories: {cats}")
        print(f"   Sous-categories: {subs}")
        print(f"   Segments: {segs}")
        print(f"   Test 2018+West+Furniture: CA={test[0]}, nb={test[1]}")

        expected_total = 9800
        if total == expected_total:
            print(f"\n  [OK] {total}/{expected_total} lignes inserees")
        else:
            print(f"\n  [WARN] {total} lignes inserees au lieu de {expected_total} attendues")

        expected_years = [2015, 2016, 2017, 2018]
        if years == expected_years:
            print(f"  [OK] Annees trouvees = {years}")
        else:
            print(f"  [WARN] Annees trouvees = {years}, attendues = {expected_years}")


if __name__ == "__main__":
    print("Chargement du CSV dans ventes_detail (mode batch)...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )
    try:
        create_table(conn)
        load_csv(conn)
        verify(conn)
    finally:
        conn.close()
    print("\n[OK] Termine. La table ventes_detail est prete.")
