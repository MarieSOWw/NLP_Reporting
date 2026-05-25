"""
db.py - Module de lecture centralisee PostgreSQL
==================================================

Role
----
Toute lecture de PostgreSQL par l'API, le PDF ou le chatbot RAG passe
par ce module. Source unique de verite cote lecture, identique a
save_to_postgres.py cote ecriture.

Pourquoi ce module
------------------
Sans ca, l'API, le PDF et le chatbot ecriraient chacun leurs propres
requetes SQL, avec un risque elevee de divergence (un endpoint
disant CA = X, un autre disant CA = Y). Ici, tout passe par les
memes fonctions filtrees, qui partent toutes de la table
`ventes_detail` (grain transactionnel).

Logique de filtres
------------------
Toutes les fonctions acceptent year/region/category optionnels.
Quand au moins un filtre est actif, la requete part de ventes_detail
et recalcule. Quand aucun filtre n'est actif, on retourne les
kpi_* pre-agregees par Spark (plus rapide).

Decisions canoniques
--------------------
- ANOMALIES YoY : la detection compare meme trimestre annee
  precedente. Le filtre year est applique APRES le calcul YoY pour
  preserver la reference temporelle (sinon le rapport 2017 verrait
  toujours zero anomalie). Les filtres region/category restent
  appliques avant agregation (calcul YoY sur la trajectoire propre).
- SEUIL : +/- 20% (aligne avec config.ANOMALY_THRESHOLD et
  src/nlp_transformers.detecter_anomalies).
- CROISSANCE YoY honnete : retourne None si pas d'annee N-1
  comparable, plutot que d'inventer une variation intra-annee.

Vector store
------------
La recherche semantique pour le chatbot RAG passe par ChromaDB
(client persistant dans .chroma_rag/). Les helpers
vectorstore_available() et search_rag_chunks() sont exposes ici
plutot que dans api.py pour simplifier la mock-isation en test.
"""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_CHROMA_PATH = str(PROJECT_ROOT / ".chroma_rag")
_CHROMA_COLLECTION = "rag_chunks"
_chroma_client = None
_chroma_col = None

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
except ImportError:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "nlp_reporting")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "admin123")

logger = logging.getLogger(__name__)


@contextmanager
def get_cursor():
    """Context manager : commit a la sortie, rollback en cas d'exception."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1;")
            return True
    except Exception as e:
        logger.error(f"Connexion echouee: {e}")
        return False


def _where(year=None, region=None, category=None):
    """Construit WHERE + params pour ventes_detail."""
    clauses, params = [], []
    if year:
        clauses.append("annee = %s")
        params.append(year)
    if region:
        clauses.append("region = %s")
        params.append(region)
    if category:
        clauses.append("category = %s")
        params.append(category)
    w = " WHERE " + " AND ".join(clauses) if clauses else ""
    return w, params


def _has_filters(year=None, region=None, category=None) -> bool:
    return bool(year or region or category)


def get_kpi_global() -> dict:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT ca_total, nb_commandes, panier_moyen, nb_clients,
                   croissance_globale, periode, meilleure_annee, meilleur_ca,
                   meilleure_region, nb_produits
            FROM kpi_global ORDER BY created_at DESC LIMIT 1;
            """
        )
        row = cur.fetchone()
        return dict(row) if row else {}


def get_kpi_filtered_summary(year=None, region=None, category=None) -> dict:
    """Recapitulatif filtre depuis ventes_detail. Tombe sur global si aucun filtre."""
    if not _has_filters(year, region, category):
        return get_kpi_global()

    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT
                COALESCE(SUM(sales), 0)                                     as ca_total,
                COUNT(DISTINCT order_id)                                    as nb_commandes,
                COUNT(*)                                                    as nb_articles,
                CASE WHEN COUNT(DISTINCT order_id) > 0
                     THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                     ELSE 0 END                                             as panier_moyen,
                COUNT(DISTINCT customer_id)                                 as nb_clients,
                COUNT(DISTINCT product_name)                                as nb_produits
            FROM ventes_detail {w};
            """,
            p,
        )
        row = cur.fetchone()
        if not row or not row["ca_total"]:
            return get_kpi_global()

        result = dict(row)
        result["ca_total"] = float(result["ca_total"])
        result["panier_moyen"] = float(result["panier_moyen"])

        cur.execute(
            f"""
            SELECT region, SUM(sales) as total
            FROM ventes_detail {w}
            GROUP BY region ORDER BY total DESC LIMIT 1;
            """,
            p,
        )
        best = cur.fetchone()
        result["meilleure_region"] = best["region"] if best else "N/A"

        # Croissance YoY honnete : None si pas d'annee N-1 comparable
        if year:
            prev_clauses = ["annee = %s"]
            prev_params = [year - 1]
            if region:
                prev_clauses.append("region = %s")
                prev_params.append(region)
            if category:
                prev_clauses.append("category = %s")
                prev_params.append(category)
            prev_where = " WHERE " + " AND ".join(prev_clauses)

            cur.execute(
                f"""
                WITH cur_yr AS (SELECT COALESCE(SUM(sales), 0) as ca FROM ventes_detail {w}),
                     prev_yr AS (SELECT COALESCE(SUM(sales), 0) as ca FROM ventes_detail {prev_where})
                SELECT
                    cur_yr.ca as ca_cur,
                    prev_yr.ca as ca_prev,
                    CASE WHEN prev_yr.ca > 0
                         THEN ROUND(((cur_yr.ca - prev_yr.ca) / prev_yr.ca * 100)::numeric, 1)
                         ELSE NULL END as croissance
                FROM cur_yr, prev_yr;
                """,
                p + prev_params,
            )
            yoy = cur.fetchone()

            if yoy and yoy["croissance"] is not None:
                result["croissance_globale"] = float(yoy["croissance"])
                result["croissance_type"] = "yoy"
            else:
                result["croissance_globale"] = None
                result["croissance_type"] = "n/a_first_year"

                intra_clauses = ["annee = %s"]
                intra_params = [year]
                if region:
                    intra_clauses.append("region = %s")
                    intra_params.append(region)
                if category:
                    intra_clauses.append("category = %s")
                    intra_params.append(category)
                intra_where = " WHERE " + " AND ".join(intra_clauses)

                cur.execute(
                    f"""
                    SELECT trimestre, COALESCE(SUM(sales), 0) as ca
                    FROM ventes_detail {intra_where}
                    GROUP BY trimestre ORDER BY trimestre;
                    """,
                    intra_params,
                )
                qtrs = cur.fetchall()
                if qtrs and len(qtrs) >= 2:
                    ca_t1 = float(qtrs[0]["ca"])
                    ca_last = float(qtrs[-1]["ca"])
                    if ca_t1 > 0:
                        result["variation_intra_annee_qoq"] = round(
                            (ca_last - ca_t1) / ca_t1 * 100, 1
                        )
                    else:
                        result["variation_intra_annee_qoq"] = None
                else:
                    result["variation_intra_annee_qoq"] = None
        else:
            g = get_kpi_global()
            result["croissance_globale"] = float(g.get("croissance_globale", 0) or 0)
            result["croissance_type"] = "yoy"

        result["periode"] = str(year) if year else "2015-2018"
        result["meilleure_annee"] = year or get_kpi_global().get("meilleure_annee")
        result["meilleur_ca"] = result["ca_total"]
        return result


def get_kpi_annual(year=None) -> list:
    """Lecture directe de kpi_annuel (pre-agregee)."""
    with get_cursor() as cur:
        w, p = "", []
        if year:
            w = " WHERE annee = %s"
            p = [year]
        cur.execute(
            f"""
            SELECT annee, ca_annuel, nb_commandes, panier_moyen,
                   clients_uniques, croissance_yoy
            FROM kpi_annuel {w} ORDER BY annee;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_annual_filtered(year=None, region=None, category=None) -> list:
    """KPI annuels recalcules depuis ventes_detail avec filtres."""
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            WITH yearly AS (
                SELECT annee,
                       SUM(sales) as ca_annuel,
                       COUNT(DISTINCT order_id) as nb_commandes,
                       COUNT(*) as nb_articles,
                       CASE WHEN COUNT(DISTINCT order_id) > 0
                            THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                            ELSE 0 END as panier_moyen,
                       COUNT(DISTINCT customer_id) as clients_uniques
                FROM ventes_detail {w}
                GROUP BY annee
            )
            SELECT *,
                CASE WHEN LAG(ca_annuel) OVER (ORDER BY annee) > 0
                     THEN ROUND(((ca_annuel - LAG(ca_annuel) OVER (ORDER BY annee))
                                 / LAG(ca_annuel) OVER (ORDER BY annee) * 100)::numeric, 1)
                     ELSE NULL END as croissance_yoy
            FROM yearly
            ORDER BY annee;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_regions(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT region, annee, trimestre,
                   SUM(sales) as ventes_totales,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   COUNT(*) as nb_articles,
                   CASE WHEN COUNT(DISTINCT order_id) > 0
                        THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                        ELSE 0 END as panier_moyen,
                   COUNT(DISTINCT customer_id) as clients_uniques
            FROM ventes_detail {w}
            GROUP BY region, annee, trimestre
            ORDER BY annee, trimestre, region;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_regions_summary(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT region,
                   SUM(sales) as ventes_totales,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   COUNT(*) as nb_articles,
                   CASE WHEN COUNT(DISTINCT order_id) > 0
                        THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                        ELSE 0 END as panier_moyen
            FROM ventes_detail {w}
            GROUP BY region ORDER BY ventes_totales DESC;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_categories(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT region, category, annee,
                   SUM(sales) as ventes_categorie,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   COUNT(*) as nb_articles
            FROM ventes_detail {w}
            GROUP BY region, category, annee
            ORDER BY annee, ventes_categorie DESC;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_categories_summary(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT category,
                   SUM(sales) as ventes_totales,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   COUNT(*) as nb_articles
            FROM ventes_detail {w}
            GROUP BY category ORDER BY ventes_totales DESC;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_sub_categories(year=None, region=None, category=None, limit=10) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT sub_category,
                   SUM(sales) as ventes_totales,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   COUNT(*) as quantite_vendue
            FROM ventes_detail {w}
            GROUP BY sub_category
            ORDER BY ventes_totales DESC
            LIMIT %s;
            """,
            p + [limit],
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_quarterly(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            WITH quarterly AS (
                SELECT annee, trimestre, SUM(sales) as ventes_totales
                FROM ventes_detail {w}
                GROUP BY annee, trimestre
            ),
            with_prev AS (
                SELECT *,
                    LAG(ventes_totales) OVER (ORDER BY annee, trimestre) as ventes_precedentes
                FROM quarterly
            )
            SELECT annee, trimestre, ventes_totales, ventes_precedentes,
                CASE WHEN ventes_precedentes > 0
                     THEN ROUND(((ventes_totales - ventes_precedentes) / ventes_precedentes * 100)::numeric, 1)
                     ELSE NULL
                END as variation_pct,
                ROUND(((SUM(ventes_totales) OVER (ORDER BY annee, trimestre) /
                    NULLIF(FIRST_VALUE(ventes_totales) OVER (ORDER BY annee, trimestre), 0)) - 1) * 100, 1)::numeric
                    as croissance_cumul
            FROM with_prev
            ORDER BY annee, trimestre;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_monthly(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            WITH monthly AS (
                SELECT annee, mois,
                       SUM(sales) as ventes_mensuelles,
                       COUNT(DISTINCT order_id) as nb_commandes,
                       CASE WHEN COUNT(DISTINCT order_id) > 0
                            THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                            ELSE 0 END as panier_moyen
                FROM ventes_detail {w}
                GROUP BY annee, mois
            )
            SELECT annee, mois, ventes_mensuelles, nb_commandes, panier_moyen,
                ROUND(AVG(ventes_mensuelles) OVER (
                    ORDER BY annee, mois ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                )::numeric, 2) as moyenne_mobile_3m,
                TO_DATE(annee || '-' || LPAD(mois::text, 2, '0') || '-01', 'YYYY-MM-DD') as date_mois
            FROM monthly
            ORDER BY annee, mois;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_segments(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT segment, annee,
                   SUM(sales) as ventes_segment,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   CASE WHEN COUNT(DISTINCT order_id) > 0
                        THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                        ELSE 0 END as panier_moyen,
                   COUNT(DISTINCT customer_id) as nb_clients
            FROM ventes_detail {w}
            GROUP BY segment, annee
            ORDER BY annee, ventes_segment DESC;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_segments_summary(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT segment,
                   SUM(sales) as ventes_totales,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   CASE WHEN COUNT(DISTINCT order_id) > 0
                        THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                        ELSE 0 END as panier_moyen
            FROM ventes_detail {w}
            GROUP BY segment ORDER BY ventes_totales DESC;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_kpi_top_products(year=None, region=None, category=None) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT sub_category,
                   SUM(sales) as ventes_totales,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   COUNT(*) as quantite_vendue,
                   CASE WHEN COUNT(DISTINCT order_id) > 0
                        THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                        ELSE 0 END as panier_moyen,
                   COUNT(DISTINCT customer_id) as nb_clients
            FROM ventes_detail {w}
            GROUP BY sub_category
            ORDER BY ventes_totales DESC
            LIMIT 10;
            """,
            p,
        )
        return [dict(r) for r in cur.fetchall()]


def get_top_clients(year=None, region=None, category=None, limit=10) -> list:
    w, p = _where(year, region, category)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT customer_id,
                   SUM(sales) as ca_total,
                   COUNT(DISTINCT order_id) as nb_commandes,
                   CASE WHEN COUNT(DISTINCT order_id) > 0
                        THEN ROUND((SUM(sales) / COUNT(DISTINCT order_id))::numeric, 2)
                        ELSE 0 END as panier_moyen,
                   MAX(order_date) as derniere_commande
            FROM ventes_detail {w}
            GROUP BY customer_id
            ORDER BY ca_total DESC
            LIMIT %s;
            """,
            p + [limit],
        )
        return [dict(r) for r in cur.fetchall()]


def get_anomalies(year=None, region=None, category=None) -> list:
    """
    Anomalies YoY (+/-20%) avec reference temporelle preservee.

    Le filtre `year` est applique APRES le calcul du LAG, pour que
    le rapport annee N puisse comparer ses trimestres a ceux de N-1.
    Les filtres `region` / `category` restent appliques avant agregation
    (calcul YoY sur la trajectoire propre de la region/categorie).
    """
    pre_clauses, pre_params = [], []
    if region:
        pre_clauses.append("region = %s")
        pre_params.append(region)
    if category:
        pre_clauses.append("category = %s")
        pre_params.append(category)
    pre_where = " WHERE " + " AND ".join(pre_clauses) if pre_clauses else ""

    post_where = ""
    post_params = []
    if year:
        post_where = " AND annee = %s"
        post_params = [year]

    with get_cursor() as cur:
        cur.execute(
            f"""
            WITH quarterly AS (
                SELECT annee, trimestre, SUM(sales) as ventes_totales
                FROM ventes_detail {pre_where}
                GROUP BY annee, trimestre
            ),
            with_var AS (
                SELECT *,
                    LAG(ventes_totales) OVER (
                        PARTITION BY trimestre ORDER BY annee
                    ) as prev,
                    CASE WHEN LAG(ventes_totales) OVER (
                        PARTITION BY trimestre ORDER BY annee
                    ) > 0
                         THEN ROUND((
                            (ventes_totales - LAG(ventes_totales) OVER (
                                PARTITION BY trimestre ORDER BY annee
                            ))
                            / LAG(ventes_totales) OVER (
                                PARTITION BY trimestre ORDER BY annee
                            ) * 100
                         )::numeric, 1)
                         ELSE NULL
                    END as variation
                FROM quarterly
            )
            SELECT
                CASE WHEN variation < -20 THEN 'ALERTE' ELSE 'INFO' END as niveau,
                CASE WHEN variation < 0 THEN 'baisse' ELSE 'hausse' END as type_anomalie,
                annee, trimestre, variation, ventes_totales as ventes,
                CASE
                    WHEN variation < -20 THEN
                        'ALERTE - Chute YoY de ' || ABS(variation) || chr(37) || ' au T' || trimestre || ' ' || annee
                        || ' vs T' || trimestre || ' ' || (annee - 1)
                        || ' (' || TRIM(TO_CHAR(ventes_totales, '999,999,999')) || ' USD). Analyse recommandee.'
                    WHEN variation > 20 THEN
                        'INFO - Hausse YoY de +' || variation || chr(37) || ' au T' || trimestre || ' ' || annee
                        || ' vs T' || trimestre || ' ' || (annee - 1)
                        || ' (' || TRIM(TO_CHAR(ventes_totales, '999,999,999')) || ' USD).'
                    ELSE NULL
                END as description
            FROM with_var
            WHERE variation IS NOT NULL
              AND (variation < -20 OR variation > 20)
              {post_where}
            ORDER BY
                CASE WHEN variation < -20 THEN 0 ELSE 1 END,
                ABS(variation) DESC;
            """,
            pre_params + post_params,
        )
        return [dict(r) for r in cur.fetchall()]


def get_nlp_report(report_type="global", lang="fr",
                   year=None, region=None, category=None) -> dict:
    """Lit le rapport NLP cache + score_nlp persiste (detection auto des colonnes)."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name='rapports_nlp';
            """
        )
        cols = {r["column_name"] for r in cur.fetchall()}
        has_score_nlp = "score_nlp" in cols
        score_nlp_cols = (
            ", score_nlp, score_nlp_mention, score_nlp_details, score_nlp_lacunes"
            if has_score_nlp else ""
        )

        query = f"""
            SELECT langue, rapport_complet, resume_bullet1,
                   resume_bullet2, resume_bullet3, score, mention,
                   periode, croissance_pct, report_type,
                   filter_year, filter_region, filter_category,
                   generated_at
                   {score_nlp_cols}
            FROM rapports_nlp
            WHERE langue = %s AND report_type = %s
        """
        params = [lang, report_type]
        if year:
            query += " AND filter_year = %s"
            params.append(year)
        if region:
            query += " AND filter_region = %s"
            params.append(region)
        if category:
            query += " AND filter_category = %s"
            params.append(category)
        query += " ORDER BY generated_at DESC LIMIT 1;"
        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else {}


def get_nlp_reports_list() -> list:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, langue, report_type, filter_year, filter_region,
                   filter_category, score, mention, generated_at
            FROM rapports_nlp ORDER BY generated_at DESC;
            """
        )
        return [dict(r) for r in cur.fetchall()]


def get_rapport_structure(report_type: str = "global",
                          year: int = None,
                          region: str = None,
                          category: str = None) -> dict:
    """Structure JSON validee (recommandations hierarchisees). {} si absente."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name='rapports_nlp' AND column_name='structure_json';
            """
        )
        if not cur.fetchone():
            return {}
        cur.execute(
            """
            SELECT structure_json
            FROM rapports_nlp
            WHERE report_type = %s
              AND filter_year IS NOT DISTINCT FROM %s
              AND filter_region IS NOT DISTINCT FROM %s
              AND filter_category IS NOT DISTINCT FROM %s
            ORDER BY generated_at DESC
            LIMIT 1;
            """,
            (report_type, year, region, category),
        )
        row = cur.fetchone()
        if row and row.get("structure_json"):
            return row["structure_json"]
        return {}


def get_nltk_analysis(report_type: str = "global",
                      year: int = None,
                      region: str = None,
                      category: str = None) -> dict:
    """Analyse NLTK du rapport Mistral. Mode degrade si schema legacy."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name='nltk_analysis';
            """
        )
        cols = {r["column_name"] for r in cur.fetchall()}
        has_v55 = "couverture_score" in cols and "report_type" in cols

        if has_v55:
            cur.execute(
                """
                SELECT report_type, filter_year, filter_region, filter_category,
                       nb_phrases, nb_mots,
                       sentiment_pos, sentiment_neg, sentiment_neu, sentiment_score,
                       keywords, couverture_score, tonalite,
                       nb_classifs_tendance, nb_classifs_risque,
                       nb_classifs_recommandation, themes_dominants,
                       analyzed_at
                FROM nltk_analysis
                WHERE report_type = %s
                  AND filter_year IS NOT DISTINCT FROM %s
                  AND filter_region IS NOT DISTINCT FROM %s
                  AND filter_category IS NOT DISTINCT FROM %s
                ORDER BY analyzed_at DESC LIMIT 1;
                """,
                (report_type, year, region, category),
            )
        else:
            ts_col = "analyzed_at" if "analyzed_at" in cols else "created_at"
            cur.execute(
                f"""
                SELECT nb_phrases, nb_mots, mots_uniques,
                       sentiment_pos, sentiment_neg, sentiment_neu, sentiment_score,
                       keywords, {ts_col} as analyzed_at
                FROM nltk_analysis ORDER BY {ts_col} DESC LIMIT 1;
                """
            )
        row = cur.fetchone()
        return dict(row) if row else {}


def _get_chroma_collection():
    """Singleton ChromaDB (lazy init)."""
    global _chroma_client, _chroma_col
    if _chroma_col is not None:
        return _chroma_col
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _chroma_col = _chroma_client.get_collection(_CHROMA_COLLECTION)
        return _chroma_col
    except Exception:
        return None


def vectorstore_available() -> bool:
    """True si ChromaDB est disponible et contient des chunks."""
    col = _get_chroma_collection()
    if col is None:
        return False
    try:
        return col.count() > 0
    except Exception:
        return False


pgvector_available = vectorstore_available  # alias retrocompatible


def search_rag_chunks(query_embedding: list, top_k: int = 5,
                      report_type: str = None, year: int = None,
                      region: str = None, category: str = None) -> list:
    """Top-K chunks les plus similaires via ChromaDB cosine distance."""
    col = _get_chroma_collection()
    if col is None:
        return []
    try:
        where_conditions = []
        if report_type:
            where_conditions.append({"report_type": {"$eq": report_type}})
        if year:
            where_conditions.append({"filter_year": {"$eq": year}})
        if region:
            where_conditions.append({"filter_region": {"$eq": region}})
        if category:
            where_conditions.append({"filter_category": {"$eq": category}})

        where = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, col.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
        chunks = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance
                chunks.append({
                    "id": doc_id,
                    "report_type":     meta.get("report_type", ""),
                    "filter_year":     meta.get("filter_year") if meta.get("filter_year", -1) != -1 else None,
                    "filter_region":   meta.get("filter_region") or None,
                    "filter_category": meta.get("filter_category") or None,
                    "chunk_type":      meta.get("chunk_type", ""),
                    "content":         results["documents"][0][i],
                    "similarity":      round(similarity, 4),
                })
        return chunks
    except Exception:
        return []


def get_rag_stats() -> dict:
    col = _get_chroma_collection()
    if col is None:
        return {"available": False, "total": 0,
                "note": "ChromaDB non initialise. Lance load_rag_chunks.py"}
    try:
        total = col.count()
        if total == 0:
            return {"available": False, "total": 0,
                    "note": "Aucun chunk indexe. Lance load_rag_chunks.py"}
        results = col.get(include=["metadatas"])
        by_type = {}
        for meta in results.get("metadatas", []):
            t = meta.get("chunk_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "available": True,
            "total": total,
            "by_type": by_type,
            "backend": "chromadb",
            "path": _CHROMA_PATH,
        }
    except Exception as e:
        return {"available": False, "total": 0, "error": str(e)}


def get_filters_metadata() -> dict:
    with get_cursor() as cur:
        cur.execute("SELECT DISTINCT annee FROM ventes_detail ORDER BY annee;")
        years = [r["annee"] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT region FROM ventes_detail ORDER BY region;")
        regions = [r["region"] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT category FROM ventes_detail ORDER BY category;")
        categories = [r["category"] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT segment FROM ventes_detail ORDER BY segment;")
        segments = [r["segment"] for r in cur.fetchall()]
        return {
            "years": years,
            "regions": regions,
            "categories": categories,
            "segments": segments,
        }


def get_tables_stats() -> dict:
    tables = [
        "kpi_global", "kpi_annuel", "kpi_region_trim", "kpi_variation",
        "kpi_top_produits", "kpi_mensuel", "kpi_segment", "kpi_categorie",
        "anomalies", "rapports_nlp", "nltk_analysis", "ventes_detail",
        "rag_chunks",
    ]
    stats = {}
    with get_cursor() as cur:
        for t in tables:
            try:
                cur.execute(f"SELECT COUNT(*) as n FROM {t};")
                stats[t] = cur.fetchone()["n"]
            except Exception:
                stats[t] = "?"
    return stats
