"""
save_to_postgres.py - Export des KPIs et rapports vers PostgreSQL
====================================================================

Role
----
Cinquieme maillon du pipeline. Persiste dans PostgreSQL :
- les 12 KPIs Spark agreges (tables kpi_*)
- la table de detail granulaire (ventes_detail, chargee via
  load_detail_table.py)
- les rapports NLP generes par Mistral (rapports_nlp)
- les anomalies detectees (anomalies)
- l'analyse NLTK du rapport (nltk_analysis)

Pourquoi tout passer par PostgreSQL
-----------------------------------
- Source unique de verite. Aucun composant aval (Grafana, API, PDF)
  ne recalcule un KPI : chacun lit la meme valeur, donc les chiffres
  affiches sont coherents partout.
- Permet de cacher les rapports NLP entre runs (1 par filtre).
- Permet a Grafana de brancher directement (SQL natif).

Architecture
------------
11 tables :
    kpi_global, kpi_annuel, kpi_region_trim, kpi_variation,
    kpi_top_produits, kpi_mensuel, kpi_segment, kpi_categorie,
    rapports_nlp, anomalies, nltk_analysis
    (+ ventes_detail via load_detail_table.py)

Migration douce
---------------
creer_tables() applique des ALTER TABLE ADD COLUMN conditionnels pour
ajouter de nouvelles colonnes sans DROP. Ce qui permet de tourner sur
une vieille base sans la casser.

Decisions canoniques
--------------------
- _get(row, *keys) : accesseur resilient aux Row Spark. Permet de
  chercher plusieurs noms de colonnes pour gerer la retrocompat
  entre versions du DataFrame (ex : Ventes_Precedentes_QoQ vs
  Ventes_Precedentes).
- L'analyse NLTK est faite sur le VRAI rapport Mistral genere, pas
  sur un template. La fonction sauvegarder_nltk_analysis() est
  appelee APRES la generation par main.py, une fois par rapport.
- TRUNCATE RESTART IDENTITY CASCADE avant chaque insertion de KPI :
  on remplace, on n'accumule pas.

Fonctions publiques
-------------------
- get_connection()                    : psycopg2 connection
- creer_tables(conn)                  : create + migration douce
- sauvegarder_kpi_global(conn, k, f)  : 1 ligne dans kpi_global
- sauvegarder_kpis(conn, kpis)        : insertion des 7 tables kpi_*
- sauvegarder_rapport(conn, r, f, *)  : rapport NLP + structure JSON
- sauvegarder_nltk_analysis(...)      : analyse NLTK du rapport
- sauvegarder_tout(...)               : pipeline complet (KPI + rapport global)
- verifier_insertion(conn)            : affiche le nb de lignes par table
"""

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from psycopg2.extras import execute_values

try:
    from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
except ImportError:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "nlp_reporting")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "admin123")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get(row, *keys, default=None):
    """Accesseur resilient. Cherche plusieurs noms dans une Row Spark."""
    for k in keys:
        try:
            v = row[k]
            if v is not None:
                return v
        except (KeyError, TypeError, ValueError):
            continue
    return default


def get_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=int(DB_PORT), dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
        )
        conn.autocommit = False
        logger.info(f"[OK] Connecte a PostgreSQL: {DB_NAME}@{DB_HOST}:{DB_PORT}")
        return conn
    except psycopg2.OperationalError as e:
        logger.error("[KO] Connexion PostgreSQL impossible:")
        logger.error(f"   Host: {DB_HOST}:{DB_PORT}, DB: {DB_NAME}, User: {DB_USER}")
        logger.error(f"   Erreur: {e}")
        raise


def creer_tables(conn):
    """Cree les 11 tables + applique la migration douce des colonnes recentes."""
    sql = """
    CREATE TABLE IF NOT EXISTS kpi_global (
        id                  SERIAL PRIMARY KEY,
        ca_total            NUMERIC(14,2),
        nb_commandes        INTEGER,
        nb_articles         INTEGER,
        panier_moyen        NUMERIC(10,2),
        nb_clients          INTEGER,
        nb_produits         INTEGER,
        croissance_globale  NUMERIC(8,2),
        periode             VARCHAR(20),
        meilleure_annee     INTEGER,
        meilleur_ca         NUMERIC(14,2),
        meilleure_region    VARCHAR(50),
        created_at          TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS kpi_annuel (
        id              SERIAL PRIMARY KEY,
        annee           INTEGER NOT NULL,
        ca_annuel       NUMERIC(14,2),
        nb_commandes    INTEGER,
        nb_articles     INTEGER,
        panier_moyen    NUMERIC(10,2),
        clients_uniques INTEGER,
        produits_vendus INTEGER,
        croissance_yoy  NUMERIC(8,2),
        created_at      TIMESTAMP DEFAULT NOW(),
        UNIQUE(annee)
    );

    CREATE TABLE IF NOT EXISTS kpi_region_trim (
        id              SERIAL PRIMARY KEY,
        region          VARCHAR(50),
        annee           INTEGER,
        trimestre       INTEGER,
        ventes_totales  NUMERIC(14,2),
        nb_commandes    INTEGER,
        nb_articles     INTEGER,
        panier_moyen    NUMERIC(10,2),
        clients_uniques INTEGER,
        created_at      TIMESTAMP DEFAULT NOW(),
        UNIQUE(region, annee, trimestre)
    );

    CREATE TABLE IF NOT EXISTS kpi_variation (
        id                  SERIAL PRIMARY KEY,
        annee               INTEGER,
        trimestre           INTEGER,
        ventes_totales      NUMERIC(14,2),
        ventes_precedentes  NUMERIC(14,2),
        variation_pct       NUMERIC(8,2),
        croissance_cumul    NUMERIC(8,2),
        created_at          TIMESTAMP DEFAULT NOW(),
        UNIQUE(annee, trimestre)
    );

    CREATE TABLE IF NOT EXISTS kpi_top_produits (
        id                   SERIAL PRIMARY KEY,
        sub_category         VARCHAR(100),
        ventes_totales       NUMERIC(14,2),
        nb_commandes         INTEGER,
        quantite_vendue      INTEGER,
        prix_moyen_article   NUMERIC(10,2),
        nb_clients           INTEGER,
        created_at           TIMESTAMP DEFAULT NOW(),
        UNIQUE(sub_category)
    );

    CREATE TABLE IF NOT EXISTS kpi_mensuel (
        id                  SERIAL PRIMARY KEY,
        annee               INTEGER,
        mois                INTEGER,
        ventes_mensuelles   NUMERIC(14,2),
        nb_commandes        INTEGER,
        panier_moyen        NUMERIC(10,2),
        moyenne_mobile_3m   NUMERIC(14,2),
        date_mois           DATE,
        created_at          TIMESTAMP DEFAULT NOW(),
        UNIQUE(annee, mois)
    );

    CREATE TABLE IF NOT EXISTS kpi_segment (
        id              SERIAL PRIMARY KEY,
        segment         VARCHAR(50),
        annee           INTEGER,
        ventes_segment  NUMERIC(14,2),
        nb_commandes    INTEGER,
        panier_moyen    NUMERIC(10,2),
        nb_clients      INTEGER,
        created_at      TIMESTAMP DEFAULT NOW(),
        UNIQUE(segment, annee)
    );

    CREATE TABLE IF NOT EXISTS kpi_categorie (
        id                  SERIAL PRIMARY KEY,
        region              VARCHAR(50),
        category            VARCHAR(100),
        annee               INTEGER,
        ventes_categorie    NUMERIC(14,2),
        nb_articles         INTEGER,
        prix_moyen_article  NUMERIC(10,2),
        created_at          TIMESTAMP DEFAULT NOW(),
        UNIQUE(region, category, annee)
    );

    CREATE TABLE IF NOT EXISTS rapports_nlp (
        id              SERIAL PRIMARY KEY,
        langue          VARCHAR(10),
        report_type     VARCHAR(20) DEFAULT 'global',
        filter_year     INTEGER,
        filter_region   VARCHAR(50),
        filter_category VARCHAR(100),
        rapport_complet TEXT,
        resume_bullet1  TEXT,
        resume_bullet2  TEXT,
        resume_bullet3  TEXT,
        score           INTEGER,
        mention         VARCHAR(50),
        periode         VARCHAR(20),
        croissance_pct  NUMERIC(6,1),
        structure_json  JSONB,
        prompt_version  VARCHAR(20),
        score_nlp        INTEGER,
        score_nlp_mention VARCHAR(50),
        score_nlp_details JSONB,
        score_nlp_lacunes JSONB,
        generated_at    TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS anomalies (
        id            SERIAL PRIMARY KEY,
        niveau        VARCHAR(10),
        type_anomalie VARCHAR(20),
        annee         INTEGER,
        trimestre     INTEGER,
        variation     NUMERIC(8,2),
        ventes        NUMERIC(14,2),
        description   TEXT,
        created_at    TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS nltk_analysis (
        id              SERIAL PRIMARY KEY,
        report_type     VARCHAR(20) DEFAULT 'global',
        filter_year     INTEGER,
        filter_region   VARCHAR(50),
        filter_category VARCHAR(100),
        nb_phrases      INTEGER,
        nb_mots         INTEGER,
        mots_uniques    INTEGER,
        sentiment_pos   NUMERIC(5,4),
        sentiment_neg   NUMERIC(5,4),
        sentiment_neu   NUMERIC(5,4),
        sentiment_score NUMERIC(5,4),
        keywords        TEXT,
        couverture_score INTEGER,
        tonalite        VARCHAR(20),
        nb_classifs_tendance       INTEGER,
        nb_classifs_risque         INTEGER,
        nb_classifs_recommandation INTEGER,
        themes_dominants TEXT,
        analyzed_at     TIMESTAMP DEFAULT NOW()
    );
    """

    migration_sql = """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='kpi_global' AND column_name='nb_articles') THEN
            ALTER TABLE kpi_global ADD COLUMN nb_articles INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='kpi_annuel' AND column_name='nb_articles') THEN
            ALTER TABLE kpi_annuel ADD COLUMN nb_articles INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='kpi_annuel' AND column_name='produits_vendus') THEN
            ALTER TABLE kpi_annuel ADD COLUMN produits_vendus INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='kpi_region_trim' AND column_name='nb_articles') THEN
            ALTER TABLE kpi_region_trim ADD COLUMN nb_articles INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='kpi_top_produits' AND column_name='nb_commandes') THEN
            ALTER TABLE kpi_top_produits ADD COLUMN nb_commandes INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='kpi_top_produits' AND column_name='prix_moyen_article') THEN
            ALTER TABLE kpi_top_produits ADD COLUMN prix_moyen_article NUMERIC(10,2);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='kpi_categorie' AND column_name='nb_articles') THEN
            ALTER TABLE kpi_categorie ADD COLUMN nb_articles INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='kpi_categorie' AND column_name='prix_moyen_article') THEN
            ALTER TABLE kpi_categorie ADD COLUMN prix_moyen_article NUMERIC(10,2);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='rapports_nlp' AND column_name='structure_json') THEN
            ALTER TABLE rapports_nlp ADD COLUMN structure_json JSONB;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='rapports_nlp' AND column_name='prompt_version') THEN
            ALTER TABLE rapports_nlp ADD COLUMN prompt_version VARCHAR(20);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='rapports_nlp' AND column_name='score_nlp') THEN
            ALTER TABLE rapports_nlp ADD COLUMN score_nlp INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='rapports_nlp' AND column_name='score_nlp_mention') THEN
            ALTER TABLE rapports_nlp ADD COLUMN score_nlp_mention VARCHAR(50);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='rapports_nlp' AND column_name='score_nlp_details') THEN
            ALTER TABLE rapports_nlp ADD COLUMN score_nlp_details JSONB;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='rapports_nlp' AND column_name='score_nlp_lacunes') THEN
            ALTER TABLE rapports_nlp ADD COLUMN score_nlp_lacunes JSONB;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='report_type') THEN
            ALTER TABLE nltk_analysis ADD COLUMN report_type VARCHAR(20) DEFAULT 'global';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='filter_year') THEN
            ALTER TABLE nltk_analysis ADD COLUMN filter_year INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='filter_region') THEN
            ALTER TABLE nltk_analysis ADD COLUMN filter_region VARCHAR(50);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='filter_category') THEN
            ALTER TABLE nltk_analysis ADD COLUMN filter_category VARCHAR(100);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='couverture_score') THEN
            ALTER TABLE nltk_analysis ADD COLUMN couverture_score INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='tonalite') THEN
            ALTER TABLE nltk_analysis ADD COLUMN tonalite VARCHAR(20);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='nb_classifs_tendance') THEN
            ALTER TABLE nltk_analysis ADD COLUMN nb_classifs_tendance INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='nb_classifs_risque') THEN
            ALTER TABLE nltk_analysis ADD COLUMN nb_classifs_risque INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='nb_classifs_recommandation') THEN
            ALTER TABLE nltk_analysis ADD COLUMN nb_classifs_recommandation INTEGER;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='nltk_analysis' AND column_name='themes_dominants') THEN
            ALTER TABLE nltk_analysis ADD COLUMN themes_dominants TEXT;
        END IF;
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='nltk_analysis' AND column_name='created_at')
           AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name='nltk_analysis' AND column_name='analyzed_at') THEN
            ALTER TABLE nltk_analysis RENAME COLUMN created_at TO analyzed_at;
        END IF;
    END $$;

    CREATE INDEX IF NOT EXISTS idx_rapports_nlp_lookup
        ON rapports_nlp (report_type, filter_year, filter_region, filter_category);
    CREATE INDEX IF NOT EXISTS idx_nltk_analysis_lookup
        ON nltk_analysis (report_type, filter_year, filter_region, filter_category);
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(migration_sql)
    conn.commit()
    logger.info("[OK] Tables creees/migrees (11 tables)")


def sauvegarder_kpi_global(conn, kpis: dict, faits: dict):
    """Sauvegarde le recap global. Recalcule nb_clients depuis ventes_detail si dispo."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE kpi_global RESTART IDENTITY CASCADE;")

        annuels = kpis.get("annuel", [])
        if not annuels:
            logger.warning("Pas de KPIs annuels, kpi_global ignore")
            return

        ca_total = sum(float(_get(r, "CA_Annuel", default=0)) for r in annuels)
        nb_commandes = sum(int(_get(r, "Nb_Commandes", default=0)) for r in annuels)
        nb_articles = sum(int(_get(r, "Nb_Articles", default=0)) for r in annuels)
        panier_moyen = round(ca_total / nb_commandes, 2) if nb_commandes else 0
        nb_clients = max(int(_get(r, "Clients_Uniques", default=0)) for r in annuels)

        best = max(annuels, key=lambda r: float(_get(r, "CA_Annuel", default=0)))
        meilleure_annee = int(_get(best, "Annee", default=0))
        meilleur_ca = float(_get(best, "CA_Annuel", default=0))

        nb_produits = max(
            (int(_get(r, "Produits_Vendus", default=0)) for r in annuels),
            default=0,
        )

        try:
            cur.execute("SELECT COUNT(DISTINCT customer_id) FROM ventes_detail;")
            real_clients = cur.fetchone()
            if real_clients and real_clients[0]:
                nb_clients = int(real_clients[0])
                logger.info(f"   nb_clients corrige depuis ventes_detail: {nb_clients}")
        except Exception:
            pass

        regions = kpis.get("region_trim", [])
        region_totals = {}
        for r in regions:
            reg = _get(r, "Region", default="N/A")
            region_totals[reg] = region_totals.get(reg, 0) + float(_get(r, "Ventes_Totales", default=0))
        meilleure_region = max(region_totals, key=region_totals.get) if region_totals else "N/A"

        croissance = faits.get("croissance_globale", 0)
        periode = faits.get("periode", "N/A")

        cur.execute(
            """
            INSERT INTO kpi_global
                (ca_total, nb_commandes, nb_articles, panier_moyen, nb_clients,
                 nb_produits, croissance_globale, periode, meilleure_annee,
                 meilleur_ca, meilleure_region)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                ca_total, nb_commandes, nb_articles, panier_moyen, nb_clients,
                nb_produits, croissance, periode, meilleure_annee,
                meilleur_ca, meilleure_region,
            ),
        )
        logger.info(
            f"   kpi_global: CA={ca_total:,.0f} | "
            f"Cmd={nb_commandes:,} | Articles={nb_articles:,} | Panier=${panier_moyen}"
        )

    conn.commit()


def sauvegarder_kpis(conn, kpis: dict):
    """Sauvegarde les 7 tables kpi_* a partir des KPIs Spark collectees."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE kpi_annuel RESTART IDENTITY CASCADE;")
        rows = []
        for r in kpis["annuel"]:
            rows.append((
                int(_get(r, "Annee")),
                float(_get(r, "CA_Annuel", default=0)),
                int(_get(r, "Nb_Commandes", default=0)),
                int(_get(r, "Nb_Articles", default=0)),
                float(_get(r, "Panier_Moyen", default=0)),
                int(_get(r, "Clients_Uniques", default=0)),
                int(_get(r, "Produits_Vendus", default=0)),
                float(_get(r, "Croissance_YoY")) if _get(r, "Croissance_YoY") is not None else None,
            ))
        execute_values(
            cur,
            """INSERT INTO kpi_annuel
               (annee, ca_annuel, nb_commandes, nb_articles, panier_moyen,
                clients_uniques, produits_vendus, croissance_yoy)
               VALUES %s""",
            rows,
        )
        logger.info(f"   kpi_annuel: {len(rows)} lignes")

        cur.execute("TRUNCATE TABLE kpi_region_trim RESTART IDENTITY CASCADE;")
        rows = [(
            _get(r, "Region"),
            int(_get(r, "Annee")),
            int(_get(r, "Trimestre")),
            float(_get(r, "Ventes_Totales", default=0)),
            int(_get(r, "Nb_Commandes", default=0)),
            int(_get(r, "Nb_Articles", default=0)),
            float(_get(r, "Panier_Moyen", default=0)),
            int(_get(r, "Clients_Uniques", default=0)),
        ) for r in kpis["region_trim"]]
        execute_values(
            cur,
            """INSERT INTO kpi_region_trim
               (region, annee, trimestre, ventes_totales, nb_commandes,
                nb_articles, panier_moyen, clients_uniques)
               VALUES %s""",
            rows,
        )
        logger.info(f"   kpi_region_trim: {len(rows)} lignes")

        cur.execute("TRUNCATE TABLE kpi_variation RESTART IDENTITY CASCADE;")
        rows = [(
            int(_get(r, "Annee")),
            int(_get(r, "Trimestre")),
            float(_get(r, "Ventes_Totales", default=0)),
            float(_get(r, "Ventes_Precedentes_QoQ", "Ventes_Precedentes"))
                if _get(r, "Ventes_Precedentes_QoQ", "Ventes_Precedentes") is not None else None,
            float(_get(r, "Variation_Pct")) if _get(r, "Variation_Pct") is not None else None,
            float(_get(r, "Croissance_Cumul")) if _get(r, "Croissance_Cumul") is not None else None,
        ) for r in kpis["variation"]]
        execute_values(
            cur,
            """INSERT INTO kpi_variation
               (annee, trimestre, ventes_totales, ventes_precedentes,
                variation_pct, croissance_cumul)
               VALUES %s""",
            rows,
        )
        logger.info(f"   kpi_variation: {len(rows)} lignes")

        cur.execute("TRUNCATE TABLE kpi_top_produits RESTART IDENTITY CASCADE;")
        rows = [(
            _get(r, "Sub_Category"),
            float(_get(r, "Ventes_Totales", default=0)),
            int(_get(r, "Nb_Commandes", default=0)),
            int(_get(r, "Quantite_Vendue", default=0)),
            float(_get(r, "Prix_Moyen_Article", "Prix_Moyen", default=0)),
            int(_get(r, "Nb_Clients", default=0)),
        ) for r in kpis["top_produits"]]
        execute_values(
            cur,
            """INSERT INTO kpi_top_produits
               (sub_category, ventes_totales, nb_commandes, quantite_vendue,
                prix_moyen_article, nb_clients)
               VALUES %s""",
            rows,
        )
        logger.info(f"   kpi_top_produits: {len(rows)} lignes")

        cur.execute("TRUNCATE TABLE kpi_mensuel RESTART IDENTITY CASCADE;")
        rows = [(
            int(_get(r, "Annee")),
            int(_get(r, "Mois")),
            float(_get(r, "Ventes_Mensuelles", default=0)),
            int(_get(r, "Nb_Commandes", default=0)),
            float(_get(r, "Panier_Moyen", default=0)),
            float(_get(r, "Moyenne_Mobile_3M")) if _get(r, "Moyenne_Mobile_3M") is not None else None,
            f"{int(_get(r, 'Annee'))}-{int(_get(r, 'Mois')):02d}-01",
        ) for r in kpis["mensuel"]]
        execute_values(
            cur,
            """INSERT INTO kpi_mensuel
               (annee, mois, ventes_mensuelles, nb_commandes, panier_moyen,
                moyenne_mobile_3m, date_mois)
               VALUES %s""",
            rows,
        )
        logger.info(f"   kpi_mensuel: {len(rows)} lignes")

        cur.execute("TRUNCATE TABLE kpi_segment RESTART IDENTITY CASCADE;")
        rows = [(
            _get(r, "Segment"),
            int(_get(r, "Annee")),
            float(_get(r, "Ventes_Segment", default=0)),
            int(_get(r, "Nb_Commandes", default=0)),
            float(_get(r, "Panier_Moyen", default=0)),
            int(_get(r, "Nb_Clients", default=0)),
        ) for r in kpis["segment"]]
        execute_values(
            cur,
            """INSERT INTO kpi_segment
               (segment, annee, ventes_segment, nb_commandes, panier_moyen, nb_clients)
               VALUES %s""",
            rows,
        )
        logger.info(f"   kpi_segment: {len(rows)} lignes")

        cur.execute("TRUNCATE TABLE kpi_categorie RESTART IDENTITY CASCADE;")
        rows = [(
            _get(r, "Region"),
            _get(r, "Category"),
            int(_get(r, "Annee")),
            float(_get(r, "Ventes_Categorie", default=0)),
            int(_get(r, "Nb_Articles", "Nb_Produits", default=0)),
            float(_get(r, "Prix_Moyen_Article", "Prix_Moyen", default=0)),
        ) for r in kpis["categorie"]]
        execute_values(
            cur,
            """INSERT INTO kpi_categorie
               (region, category, annee, ventes_categorie, nb_articles, prix_moyen_article)
               VALUES %s""",
            rows,
        )
        logger.info(f"   kpi_categorie: {len(rows)} lignes")

    conn.commit()


def sauvegarder_rapport(conn, resultats: dict, faits: dict,
                        report_type: str = "global",
                        filter_year: int = None,
                        filter_region: str = None,
                        filter_category: str = None,
                        prompt_version: str = "v6.0"):
    """Persiste un rapport NLP avec sa structure JSON et son score qualite."""
    with conn.cursor() as cur:
        bullets = resultats.get("resume", {}).get("bullets", []) or []
        structure = resultats.get("structure") or resultats.get("structure_json") or {}

        score_nlp_obj = resultats.get("score_nlp") or {}
        score_nlp_val = score_nlp_obj.get("score_nlp")
        score_nlp_mention = score_nlp_obj.get("mention", "")[:50] if score_nlp_obj.get("mention") else None
        score_nlp_details = score_nlp_obj.get("details") or {}
        score_nlp_lacunes = score_nlp_obj.get("lacunes") or []

        cur.execute(
            """
            INSERT INTO rapports_nlp
                (langue, report_type, filter_year, filter_region, filter_category,
                 rapport_complet, resume_bullet1, resume_bullet2, resume_bullet3,
                 score, mention, periode, croissance_pct,
                 structure_json, prompt_version,
                 score_nlp, score_nlp_mention, score_nlp_details, score_nlp_lacunes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                resultats.get("langue", "fr"),
                report_type,
                filter_year,
                filter_region,
                filter_category,
                resultats.get("rapport", ""),
                bullets[0] if len(bullets) > 0 else None,
                bullets[1] if len(bullets) > 1 else None,
                bullets[2] if len(bullets) > 2 else None,
                resultats.get("score", {}).get("score", 0),
                resultats.get("score", {}).get("mention", ""),
                faits.get("periode", "N/A"),
                faits.get("croissance_globale") if faits.get("croissance_globale") is not None else None,
                json.dumps(structure, ensure_ascii=False, default=str) if structure else None,
                prompt_version,
                int(score_nlp_val) if score_nlp_val is not None else None,
                score_nlp_mention,
                json.dumps(score_nlp_details, ensure_ascii=False, default=str) if score_nlp_details else None,
                json.dumps(score_nlp_lacunes, ensure_ascii=False, default=str) if score_nlp_lacunes else None,
            ),
        )
        logger.info(
            f"   rapports_nlp: 1 rapport "
            f"(type={report_type}, score={resultats.get('score', {}).get('score', 0)}/100, "
            f"score_nlp={score_nlp_val}/100, "
            f"structure={'oui' if structure else 'non'})"
        )

        if report_type == "global":
            cur.execute("TRUNCATE TABLE anomalies RESTART IDENTITY CASCADE;")
            anomalies = resultats.get("anomalies") or []
            if anomalies:
                rows = [(
                    a["niveau"],
                    a.get("type", "unknown"),
                    a["annee"],
                    a["trimestre"],
                    a["variation"],
                    a["ventes"],
                    a["fr"],
                ) for a in anomalies]
                execute_values(
                    cur,
                    """INSERT INTO anomalies
                       (niveau, type_anomalie, annee, trimestre, variation, ventes, description)
                       VALUES %s""",
                    rows,
                )
                logger.info(f"   anomalies: {len(rows)} anomalie(s)")
    conn.commit()


def sauvegarder_nltk_analysis(conn, rapport_texte: str, langue: str = "fr",
                              report_type: str = "global",
                              filter_year: int = None,
                              filter_region: str = None,
                              filter_category: str = None):
    """Analyse NLTK du rapport Mistral genere. Persiste les enrichissements business."""
    if not rapport_texte or not rapport_texte.strip():
        logger.warning(f"   nltk_analysis [{report_type}]: rapport vide, skip")
        return

    try:
        from src.nlp_nltk import analyser_rapport_genere
    except ImportError:
        try:
            from nlp_nltk import analyser_rapport_genere
        except ImportError as e:
            logger.warning(f"   nltk_analysis: import nlp_nltk impossible - {e}")
            return

    try:
        analyse = analyser_rapport_genere(rapport_texte, langue=langue)
    except Exception as e:
        logger.warning(f"   nltk_analysis [{report_type}]: echec analyse - {e}")
        return

    sentiment = analyse.get("sentiment", {}) or {}
    keywords_list = analyse.get("keywords", []) or []
    keywords = ", ".join(f"{k}:{v}" for k, v in keywords_list[:10])

    classifs = analyse.get("classifications", []) or []
    nb_tend = sum(1 for c in classifs if c.get("intent") == "tendance")
    nb_risq = sum(1 for c in classifs if c.get("intent") == "risque")
    nb_reco = sum(1 for c in classifs if c.get("intent") == "recommandation")

    couv = analyse.get("couverture", {}) or {}
    couverture_score = int(couv.get("score", 0)) if isinstance(couv.get("score"), (int, float)) else 0

    ton = analyse.get("tonalite", {}) or {}
    tonalite_label = ton.get("label", "neutre")[:20]

    themes_list = analyse.get("themes", []) or []
    themes = ", ".join(str(t) for t in themes_list[:5])

    stats = analyse.get("stats", {}) or {}
    nb_phrases = analyse.get("nb_phrases") or stats.get("nb_phrases", 0)
    nb_mots = analyse.get("nb_mots") or stats.get("nb_mots", 0)
    mots_uniques = stats.get("mots_uniques", 0)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO nltk_analysis
                (report_type, filter_year, filter_region, filter_category,
                 nb_phrases, nb_mots, mots_uniques,
                 sentiment_pos, sentiment_neg, sentiment_neu, sentiment_score,
                 keywords, couverture_score, tonalite,
                 nb_classifs_tendance, nb_classifs_risque, nb_classifs_recommandation,
                 themes_dominants)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                report_type, filter_year, filter_region, filter_category,
                int(nb_phrases or 0),
                int(nb_mots or 0),
                int(mots_uniques or 0),
                float(sentiment.get("pos", 0) or 0),
                float(sentiment.get("neg", 0) or 0),
                float(sentiment.get("neu", 0) or 0),
                float(sentiment.get("compound", 0) or 0),
                keywords,
                couverture_score,
                tonalite_label,
                nb_tend, nb_risq, nb_reco,
                themes,
            ),
        )
    conn.commit()
    logger.info(
        f"   nltk_analysis [{report_type}"
        + (f" {filter_year}" if filter_year else "")
        + (f" {filter_region}" if filter_region else "")
        + (f" {filter_category}" if filter_category else "")
        + f"]: couverture={couverture_score}/100, tonalite={tonalite_label}, "
        + f"reco={nb_reco}, risk={nb_risq}, tend={nb_tend}"
    )


def verifier_insertion(conn):
    tables = [
        "kpi_global", "kpi_annuel", "kpi_region_trim", "kpi_variation",
        "kpi_top_produits", "kpi_mensuel", "kpi_segment",
        "kpi_categorie", "rapports_nlp", "anomalies", "nltk_analysis",
    ]
    print("\n" + "=" * 55)
    print("   VERIFICATION DES TABLES POSTGRESQL")
    print("=" * 55)
    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                count = cur.fetchone()[0]
                statut = "[OK]  " if count > 0 else "[WARN]"
                print(f"  {statut} {table:<25} {count:>4} lignes")
            except Exception:
                print(f"  [KO]   {table:<25} erreur")
                conn.rollback()
    print("=" * 55)
    print(f"  Base: {DB_NAME} | Host: {DB_HOST}:{DB_PORT}")
    print("=" * 55)


def sauvegarder_tout(kpis: dict, faits: dict, resultats_nlp: dict,
                     report_type: str = "global",
                     filter_year: int = None,
                     filter_region: str = None,
                     filter_category: str = None):
    """Point d'entree complet : tables + KPI global + KPIs + rapport + analyse NLTK."""
    conn = get_connection()
    try:
        logger.info("Creation/migration des tables...")
        creer_tables(conn)

        logger.info("Sauvegarde KPI global...")
        sauvegarder_kpi_global(conn, kpis, faits)

        logger.info("Sauvegarde des KPIs Spark...")
        sauvegarder_kpis(conn, kpis)

        logger.info("Sauvegarde du rapport NLP...")
        sauvegarder_rapport(
            conn, resultats_nlp, faits,
            report_type=report_type,
            filter_year=filter_year,
            filter_region=filter_region,
            filter_category=filter_category,
        )

        logger.info("Analyse NLTK du rapport genere...")
        sauvegarder_nltk_analysis(
            conn,
            rapport_texte=resultats_nlp.get("rapport", ""),
            langue=resultats_nlp.get("langue", "fr"),
            report_type=report_type,
            filter_year=filter_year,
            filter_region=filter_region,
            filter_category=filter_category,
        )

        verifier_insertion(conn)

    except Exception as e:
        conn.rollback()
        logger.error(f"[KO] Transaction annulee: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    import warnings
    parser = argparse.ArgumentParser(description="Save to PostgreSQL")
    parser.add_argument("--langue", choices=["fr", "en"], default="fr")
    args = parser.parse_args()

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if sys.platform == "win32":
        os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME", r"C:\hadoop")
    warnings.filterwarnings("ignore")

    from src.load_data import create_spark_session, load_data
    from src.analytics import compute_kpis, collect_kpis
    from src.nlp_nltk import extraire_faits
    from src.nlp_transformers import generer_tout

    try:
        logger.info(f"Connexion a {DB_NAME} sur {DB_HOST}:{DB_PORT}...")
        spark = create_spark_session()
        df = load_data(spark)
        kpis = compute_kpis(df)
        faits = extraire_faits(kpis)
        kpis_collected = collect_kpis(kpis)
        spark.stop()
        resultats = generer_tout(faits, langue=args.langue)
        sauvegarder_tout(kpis_collected, faits, resultats)
        print("\n[OK] save_to_postgres.py termine")
        print(f"   Langue: {args.langue.upper()}")
        print(f"   Score: {resultats['score']['score']}/100 - {resultats['score']['mention']}")
    except Exception as e:
        logger.error(f"[KO] Erreur: {e}")
        raise
