"""
api.py - Serveur FastAPI pour NLP Reporting
=============================================

Role
----
Couche API REST entre PostgreSQL et les consommateurs (dashboard
HTML, PDF generator, chatbot). Sert :
    - Les KPIs business filtrables (year / region / category)
    - Les rapports NLP avec generation dynamique a la volee
    - La structure JSON des recommandations hierarchisees
    - Le score qualite NLP detaille
    - Un chatbot RAG (Mistral + ChromaDB)
    - La generation PDF a la demande

Architecture
------------
- Endpoints sous /api (configurable via config.API_PREFIX)
- Dashboard HTML servi a la racine
- Lifespan FastAPI pour logger l'etat des dependances au demarrage
- CORS ouvert par defaut (configurable via .env)

Chatbot RAG
-----------
Le chatbot utilise sentence-transformers/all-MiniLM-L6-v2 (384d) pour
embed la question utilisateur, ChromaDB pour la recherche semantique
(cosine, top 5), et Mistral pour la generation de la reponse finale.

Detection d'intentions
----------------------
Avant l'embedding, on detecte l'intention de la question via mots-cles
(top_client, top_product, anomalie, comparaison, prevision, tendance).
Cela enrichit le contexte envoye a Mistral avec les donnees pertinentes.

Anti-injection
--------------
Une liste FORBIDDEN_PATTERNS bloque les tentatives d'injection de
prompt (ignore previous, jailbreak, DAN mode, etc.). Les questions
malicieuses recoivent un refus generique.

Endpoints principaux
--------------------
GET  /api/health                Etat connexion DB et stats tables
GET  /api/filters               Metadonnees des filtres (years, regions, categories)
GET  /api/kpis/global           KPIs globaux filtrables
GET  /api/kpis/annual           Performance annuelle
GET  /api/kpis/regions          Detail regions
GET  /api/kpis/categories       Detail categories
GET  /api/kpis/quarterly        Variations trimestrielles
GET  /api/kpis/monthly          Ventes mensuelles
GET  /api/kpis/segments         Segments clients
GET  /api/kpis/top-products     Top sous-categories
GET  /api/kpis/top-clients      Top clients
GET  /api/anomalies             Anomalies YoY +/- 20%
GET  /api/nlp/report            Rapport NLP (cache ou genere live)
GET  /api/nlp/quality           Score qualite NLP detaille (Chantier 3)
GET  /api/rapport/structure     Structure JSON Chantier 1
GET  /api/nlp/analysis          Analyse NLTK enrichie (Chantier 2)
POST /api/chat                  Chatbot RAG
GET  /api/pdf/generate          Genere un PDF a la demande
POST /api/reports/schedule      Genere les 12 rapports en serie
GET  /api/dashboard             Agregat de donnees pour le frontend
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from config import (
        API_HOST, API_PORT, API_CORS_ORIGINS, API_PREFIX,
        OUTPUT_DIR, SUPPORTED_LANGUAGES, REPORT_TYPES,
        MISTRAL_API_KEY, MISTRAL_MODEL,
    )
except ImportError:
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    API_CORS_ORIGINS = ["*"]
    API_PREFIX = "/api"
    OUTPUT_DIR = Path(__file__).parent / "outputs"
    SUPPORTED_LANGUAGES = ["fr", "en"]
    REPORT_TYPES = ["global", "by_year", "by_region", "by_category"]
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")

from src.db import (
    test_connection, get_kpi_global, get_kpi_filtered_summary,
    get_kpi_annual, get_kpi_annual_filtered,
    get_kpi_regions, get_kpi_regions_summary,
    get_kpi_categories, get_kpi_categories_summary,
    get_kpi_quarterly, get_kpi_monthly,
    get_kpi_segments, get_kpi_segments_summary,
    get_kpi_top_products, get_kpi_sub_categories,
    get_top_clients, get_anomalies,
    get_nlp_report, get_nlp_reports_list,
    get_nltk_analysis, get_filters_metadata, get_tables_stats,
    vectorstore_available, search_rag_chunks, get_rag_stats,
    get_rapport_structure,
    _has_filters,
)

from src.text_utils import (
    strip_markdown as _strip_markdown,
    clean_nlp_dict as _clean_nlp,
    couleur_pour_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_faits_from_pg(year=None, region=None, category=None) -> dict:
    """
    Assemble un dict 'faits' depuis PostgreSQL filtre.
    Format identique a celui produit par nlp_nltk.extraire_faits().
    Source unique = PostgreSQL.
    """
    annual = get_kpi_annual_filtered(year=year, region=region, category=category) \
        if _has_filters(year, region, category) else get_kpi_annual()
    quarterly = get_kpi_quarterly(year=year, region=region, category=category)
    top = get_kpi_top_products(year=year, region=region, category=category)
    segments = get_kpi_segments_summary(year=year, region=region, category=category)
    summary = get_kpi_filtered_summary(year=year, region=region, category=category) \
        or get_kpi_global()

    faits_annuel = []
    for r in (annual or []):
        try:
            faits_annuel.append({
                "annee": int(r.get("annee")),
                "ca": float(r.get("ca_annuel", 0) or 0),
                "commandes": int(r.get("nb_commandes", 0) or 0),
                "panier_moyen": float(r.get("panier_moyen", 0) or 0),
                "croissance_yoy": float(r["croissance_yoy"]) if r.get("croissance_yoy") is not None else None,
            })
        except Exception:
            continue
    faits_annuel.sort(key=lambda x: x["annee"])

    if not faits_annuel:
        ca_tot = float(summary.get("ca_total", 0) or 0) if summary else 0
        nb_cmd = int(summary.get("nb_commandes", 0) or 0) if summary else 0
        faits_annuel = [{
            "annee": year or 2018, "ca": ca_tot, "commandes": nb_cmd,
            "panier_moyen": (ca_tot / nb_cmd) if nb_cmd else 0, "croissance_yoy": None,
        }]

    faits_var = []
    for r in (quarterly or []):
        try:
            var_raw = r.get("variation_pct")
            faits_var.append({
                "annee": int(r.get("annee")),
                "trimestre": int(r.get("trimestre")),
                "ventes": float(r.get("ventes_totales", 0) or 0),
                "variation": float(var_raw) if var_raw is not None else None,
            })
        except Exception:
            continue

    faits_top = []
    for r in (top or [])[:3]:
        faits_top.append({
            "nom": r.get("sub_category", "N/A"),
            "ventes": float(r.get("ventes_totales", 0) or 0),
            "qte": int(r.get("quantite_vendue", 0) or 0),
        })
    while len(faits_top) < 3:
        faits_top.append({"nom": "N/A", "ventes": 0.0, "qte": 0})

    faits_segments = {}
    for r in (segments or []):
        faits_segments[r.get("segment", "N/A")] = {
            "ventes_total": float(r.get("ventes_totales", 0) or 0),
            "nb_commandes": int(r.get("nb_commandes", 0) or 0),
        }

    meilleure_annee = max(faits_annuel, key=lambda x: x["ca"])
    pire_annee = min(faits_annuel, key=lambda x: x["ca"])

    if faits_var:
        var_calculables = [v for v in faits_var if v.get("variation") is not None]
        if var_calculables:
            meilleur_trim = max(var_calculables, key=lambda x: x["variation"])
            pire_trim = min(var_calculables, key=lambda x: x["variation"])
        else:
            meilleur_trim = max(faits_var, key=lambda x: x["ventes"])
            pire_trim = min(faits_var, key=lambda x: x["ventes"])
            meilleur_trim = dict(meilleur_trim, variation=None)
            pire_trim = dict(pire_trim, variation=None)
    else:
        meilleur_trim = {"annee": meilleure_annee["annee"], "trimestre": 1,
                         "ventes": meilleure_annee["ca"], "variation": 0.0}
        pire_trim = meilleur_trim

    ca_debut = faits_annuel[0]["ca"] or 1
    ca_fin = faits_annuel[-1]["ca"]

    if len(faits_annuel) == 1:
        yoy_sum = (summary or {}).get("croissance_globale")
        if yoy_sum is not None:
            try:
                croissance = round(float(yoy_sum), 1)
            except (TypeError, ValueError):
                croissance = None
        else:
            croissance = None
    else:
        croissance = round((ca_fin - ca_debut) / ca_debut * 100, 1) if ca_debut else 0

    annee_debut, annee_fin = faits_annuel[0]["annee"], faits_annuel[-1]["annee"]
    periode = str(annee_debut) if annee_debut == annee_fin else f"{annee_debut}-{annee_fin}"

    var_abs = [abs(v["variation"]) for v in faits_var if v.get("variation") is not None]
    vol_val = round(sum(var_abs) / len(var_abs), 1) if var_abs else 0.0
    vol_niveau = "Elevee" if vol_val > 30 else ("Moderee" if vol_val > 15 else "Faible")

    if croissance is None:
        tendance = "N/A"
    else:
        tendance = "Hausse" if croissance > 2 else ("Baisse" if croissance < -2 else "Stable")

    return {
        "periode": periode,
        "annuel": faits_annuel,
        "variations": faits_var,
        "top_produits": faits_top,
        "segments": faits_segments,
        "meilleure_annee": meilleure_annee,
        "pire_annee": pire_annee,
        "meilleur_trim": meilleur_trim,
        "pire_trim": pire_trim,
        "croissance_globale": croissance,
        "tendance_globale": {"label": tendance},
        "volatilite": {"niveau": vol_niveau, "valeur": vol_val},
        "region_focus": region,
        "categorie_focus": category,
        "regions": {},
    }


SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_sentence_transformer = None


def get_sentence_transformer():
    """Charge le Transformer HuggingFace local (singleton)."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            logger.info(f"[OK] Sentence-Transformer charge : {SENTENCE_TRANSFORMER_MODEL}")
        except Exception as e:
            logger.warning(f"Sentence-Transformers non disponible: {e}")
            _sentence_transformer = False
    return _sentence_transformer if _sentence_transformer is not False else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"NLP Reporting API starting on http://{API_HOST}:{API_PORT}")
    logger.info(f"   Mistral API: {'OK' if MISTRAL_API_KEY else 'KO'} ({MISTRAL_MODEL})")
    logger.info(f"   vectorstore (chromadb): {'OK' if vectorstore_available() else 'KO'}")
    st = get_sentence_transformer()
    logger.info(f"   Transformer HF: {'OK' if st else 'KO'}")
    logger.info("   Chantiers actifs : 1 (JSON structure) + 2 (classification) + 3 (score NLP) + 4 (recos)")
    logger.info(f"   Docs: http://{API_HOST}:{API_PORT}/docs")
    logger.info("=" * 60)
    yield
    logger.info("API arretee")


app = FastAPI(
    title="NLP Business Reporting API",
    description="API REST + Chatbot RAG (ChromaDB + Mistral + Transformer HF) + Score NLP",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/")
def root():
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"service": "NLP Business Reporting API", "version": "6.0.0"}


@app.get(f"{API_PREFIX}/health")
def health():
    db_ok = test_connection()
    stats = get_tables_stats() if db_ok else {}
    rag = get_rag_stats() if db_ok else {"available": False}
    return {
        "status": "ok" if db_ok else "error",
        "database": db_ok,
        "tables": stats,
        "rag": rag,
        "version": "6.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get(f"{API_PREFIX}/filters")
def filters():
    try:
        return get_filters_metadata()
    except Exception as e:
        raise HTTPException(500, str(e))


# KPI endpoints

@app.get(f"{API_PREFIX}/kpis/global")
def kpis_global(year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    try:
        return get_kpi_filtered_summary(year=year, region=region, category=category) or {}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/annual")
def kpis_annual(year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    try:
        if _has_filters(year, region, category):
            return get_kpi_annual_filtered(year=year, region=region, category=category)
        return get_kpi_annual()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/regions")
def kpis_regions(year: int = Query(None), region: str = Query(None),
                 category: str = Query(None), summary: bool = Query(False)):
    try:
        if summary:
            return get_kpi_regions_summary(year=year, region=region, category=category)
        return get_kpi_regions(year=year, region=region, category=category)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/categories")
def kpis_categories(year: int = Query(None), region: str = Query(None),
                    category: str = Query(None), summary: bool = Query(False)):
    try:
        if summary:
            return get_kpi_categories_summary(year=year, region=region, category=category)
        return get_kpi_categories(year=year, region=region, category=category)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/sub-categories")
def kpis_sub_categories(year: int = Query(None), region: str = Query(None),
                        category: str = Query(None), limit: int = Query(10)):
    try:
        return get_kpi_sub_categories(year=year, region=region, category=category, limit=limit)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/quarterly")
def kpis_quarterly(year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    try:
        return get_kpi_quarterly(year=year, region=region, category=category)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/monthly")
def kpis_monthly(year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    try:
        return get_kpi_monthly(year=year, region=region, category=category)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/segments")
def kpis_segments(year: int = Query(None), region: str = Query(None),
                  category: str = Query(None), summary: bool = Query(False)):
    try:
        if summary:
            return get_kpi_segments_summary(year=year, region=region, category=category)
        return get_kpi_segments(year=year, region=region, category=category)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/top-products")
def kpis_top_products(year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    try:
        return get_kpi_top_products(year=year, region=region, category=category)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/kpis/top-clients")
def kpis_top_clients(year: int = Query(None), region: str = Query(None),
                     category: str = Query(None), limit: int = Query(10)):
    try:
        return get_top_clients(year=year, region=region, category=category, limit=limit)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/anomalies")
def anomalies(year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    try:
        return get_anomalies(year=year, region=region, category=category)
    except Exception as e:
        raise HTTPException(500, str(e))


# NLP endpoints

@app.get(f"{API_PREFIX}/nlp/report")
def nlp_report(report_type: str = Query("global"), lang: str = Query("fr"),
               year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    """
    Rapport NLP : cache d'abord, sinon generation live.
    Renvoie aussi la structure JSON (Chantier 1) et le score qualite NLP (Chantier 3).
    """
    if report_type not in REPORT_TYPES:
        raise HTTPException(400, f"Type invalide. Valeurs: {REPORT_TYPES}")
    try:
        data = get_nlp_report(report_type=report_type, lang=lang,
                              year=year, region=region, category=category)
        if data and data.get("rapport_complet"):
            cleaned = _clean_nlp(data)
            try:
                struct = get_rapport_structure(report_type=report_type,
                                               year=year, region=region, category=category)
                if struct:
                    cleaned["structure"] = struct
            except Exception as e:
                logger.warning(f"get_rapport_structure failed: {e}")

            # Regenerer les bullets de resume executif a partir des faits live
            # (1 ms, garantit la coherence avec la logique courante)
            try:
                from src.nlp_transformers import generer_resume_executif
                faits_live = _build_faits_from_pg(year=year, region=region, category=category)
                resume = generer_resume_executif(faits_live, langue=lang)
                bullets_live = resume.get("bullets", []) if isinstance(resume, dict) else []
                if bullets_live:
                    cleaned["resume_bullet1"] = _strip_markdown(bullets_live[0]) \
                        if len(bullets_live) > 0 else cleaned.get("resume_bullet1", "")
                    cleaned["resume_bullet2"] = _strip_markdown(bullets_live[1]) \
                        if len(bullets_live) > 1 else cleaned.get("resume_bullet2", "")
                    cleaned["resume_bullet3"] = _strip_markdown(bullets_live[2]) \
                        if len(bullets_live) > 2 else cleaned.get("resume_bullet3", "")
                cleaned["croissance_pct"] = faits_live.get("croissance_globale")
                cleaned["periode"] = faits_live.get("periode", cleaned.get("periode", ""))
            except Exception as e:
                logger.warning(f"Regeneration bullets/croissance echouee: {e}")

            if data.get("score_nlp") is not None:
                cleaned["score_nlp"] = data.get("score_nlp")
                cleaned["score_nlp_mention"] = data.get("score_nlp_mention", "")
                cleaned["score_nlp_details"] = data.get("score_nlp_details") or {}
                cleaned["score_nlp_lacunes"] = data.get("score_nlp_lacunes") or []
            else:
                try:
                    from nlp_quality_score import evaluer_qualite_rapport
                    faits = _build_faits_from_pg(year=year, region=region, category=category)
                    score_nlp = evaluer_qualite_rapport(
                        rapport_texte=cleaned["rapport_complet"],
                        faits=faits, langue=lang,
                    )
                    cleaned["score_nlp"] = score_nlp.get("score_nlp")
                    cleaned["score_nlp_mention"] = score_nlp.get("mention")
                    cleaned["score_nlp_details"] = score_nlp.get("details")
                    cleaned["score_nlp_lacunes"] = score_nlp.get("lacunes")
                except Exception as e:
                    logger.warning(f"Score NLP non calcule sur cache: {e}")
            return cleaned

        # Generation live (aucun cache trouve pour ces filtres)
        logger.info(f"Generation NLP dynamique (year={year}, region={region}, category={category})")
        from src.nlp_transformers import (
            generer_rapport_avec_structure, generer_resume_executif, calculer_score,
        )

        faits = _build_faits_from_pg(year=year, region=region, category=category)
        res_2etapes = generer_rapport_avec_structure(faits, langue=lang)
        rapport = _strip_markdown(res_2etapes["rapport"])
        structure = res_2etapes.get("structure", {})

        try:
            resume = generer_resume_executif(faits, langue=lang)
            bullets = resume.get("bullets", []) if isinstance(resume, dict) else []
        except Exception as e:
            logger.warning(f"resume_executif failed: {e}")
            bullets = []

        try:
            score_data = calculer_score(faits)
            score_val = int(score_data.get("score", 0))
            mention = score_data.get("mention", "")
        except Exception as e:
            logger.warning(f"calculer_score failed: {e}")
            score_val, mention = 0, ""

        score_nlp = {}
        try:
            from nlp_quality_score import evaluer_qualite_rapport
            score_nlp = evaluer_qualite_rapport(rapport_texte=rapport, faits=faits, langue=lang)
        except Exception as e:
            logger.warning(f"Score NLP failed: {e}")

        return {
            "rapport_complet": rapport,
            "resume_bullet1": _strip_markdown(bullets[0]) if len(bullets) > 0 else "",
            "resume_bullet2": _strip_markdown(bullets[1]) if len(bullets) > 1 else "",
            "resume_bullet3": _strip_markdown(bullets[2]) if len(bullets) > 2 else "",
            "score": score_val,
            "mention": mention,
            "langue": lang,
            "report_type": report_type,
            "filter_year": year,
            "filter_region": region,
            "filter_category": category,
            "periode": faits.get("periode", ""),
            "croissance_pct": faits.get("croissance_globale", 0),
            "generated_at": datetime.utcnow().isoformat(),
            "source": "live",
            "structure": structure,
            "score_nlp": score_nlp.get("score_nlp"),
            "score_nlp_mention": score_nlp.get("mention"),
            "score_nlp_details": score_nlp.get("details"),
            "score_nlp_lacunes": score_nlp.get("lacunes"),
        }
    except Exception as e:
        logger.exception(f"Erreur nlp/report: {e}")
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/nlp/reports")
def nlp_reports_list():
    try:
        return get_nlp_reports_list()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/nlp/analysis")
def nlp_analysis(report_type: str = Query("global"), year: int = Query(None),
                 region: str = Query(None), category: str = Query(None)):
    """Analyse NLTK enrichie (Chantier 2) du rapport genere par Mistral."""
    try:
        data = get_nltk_analysis(report_type=report_type, year=year,
                                 region=region, category=category)
        if not data:
            raise HTTPException(404, "Aucune analyse NLTK trouvee pour ces filtres")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/nlp/business-tonalite")
def nlp_business_tonalite(report_type: str = Query("global"), lang: str = Query("fr"),
                          year: int = Query(None), region: str = Query(None),
                          category: str = Query(None)):
    """
    Tonalite business enrichie (alternative a VADER).

    Combine en un seul payload :
      - tonalite (favorable/neutre/defavorable), score_business [-100, +100]
      - niveau de risque percu (faible/modere/eleve)
      - stabilite du discours (stable/instable)
      - signaux positifs/negatifs/risque
      - themes business dominants
      - mots-cles extraits
      - couverture business (Chantier 2) si dispo
      - classifications par intention (Chantier 2) si dispo

    Calcule live depuis le rapport_complet en cache, plus enrichi avec
    les stats NLTK persistees si presentes.
    """
    try:
        rapport_data = get_nlp_report(
            report_type=report_type, lang=lang,
            year=year, region=region, category=category,
        )
        rapport_texte = rapport_data.get("rapport_complet", "") if rapport_data else ""

        if not rapport_texte:
            try:
                from src.nlp_transformers import generer_rapport
                faits_local = _build_faits_from_pg(year=year, region=region, category=category)
                rapport_texte = _strip_markdown(generer_rapport(faits_local, langue=lang))
            except Exception as e:
                logger.warning(f"Generation live rapport echouee: {e}")
                rapport_texte = ""

        if not rapport_texte:
            raise HTTPException(404, "Aucun rapport disponible pour ces filtres")

        rapport_texte = _strip_markdown(rapport_texte)

        from src.nlp_nltk import (
            evaluer_tonalite_business,
            extraire_themes_business,
            classifier_phrases_business,
            mesurer_couverture_business,
            NLTKProcessor,
        )

        tonalite = evaluer_tonalite_business(rapport_texte, langue=lang)
        themes_tuples = extraire_themes_business(rapport_texte, langue=lang, top_n=7)
        classifs = classifier_phrases_business(rapport_texte, langue=lang)
        couverture = mesurer_couverture_business(classifs)

        nltk_lang = "english" if lang == "en" else "french"
        try:
            nlp = NLTKProcessor(langue=nltk_lang)
            keywords_raw = nlp.extract_keywords(rapport_texte, top_n=10)
            stats = nlp.generate_summary_stats(rapport_texte)
        except Exception as e:
            logger.warning(f"NLTKProcessor failed: {e}")
            keywords_raw = []
            stats = {"nb_phrases": 0, "nb_mots": 0, "mots_uniques": 0}

        # Classification visuelle (4 niveaux pour le dashboard)
        score = int(tonalite.get("score_business", 0))
        niveau_risque = tonalite.get("niveau_risque", "faible")
        if niveau_risque == "eleve":
            classe_visuelle = "risque"
            classe_label = "Risque eleve" if lang == "fr" else "High risk"
        elif score >= 30:
            classe_visuelle = "favorable"
            classe_label = "Favorable" if lang == "fr" else "Favorable"
        elif score <= -20 or niveau_risque == "modere":
            classe_visuelle = "vigilance"
            classe_label = "Vigilance" if lang == "fr" else "Watch"
        else:
            classe_visuelle = "neutre"
            classe_label = "Neutre" if lang == "fr" else "Neutral"

        return {
            "report_type": report_type,
            "lang": lang,
            "filters": {"year": year, "region": region, "category": category},
            "tonalite": {
                "label": tonalite.get("tonalite", "neutre"),
                "label_lisible": tonalite.get("tonalite_label", "Neutre"),
                "classe_visuelle": classe_visuelle,
                "classe_label": classe_label,
            },
            "score_business": score,
            "niveau_risque": {
                "label": niveau_risque,
                "label_lisible": tonalite.get("niveau_risque_label", "Faible"),
            },
            "stabilite": {
                "label": tonalite.get("stabilite", "stable"),
                "label_lisible": tonalite.get("stabilite_label", "Stable"),
            },
            "signaux": {
                "positifs": int(tonalite.get("signaux_positifs", 0)),
                "negatifs": int(tonalite.get("signaux_negatifs", 0)),
                "risque": int(tonalite.get("signaux_risque", 0)),
            },
            "couverture_business": {
                "score": int(couverture.get("score_couverture", 0)),
                "mention": couverture.get("mention", ""),
                "lacunes": couverture.get("lacunes", []),
            },
            "themes_dominants": [
                {"nom": t, "score": int(s)} for t, s in themes_tuples
            ],
            "keywords": [
                {"mot": str(k), "freq": int(v)}
                for k, v in (keywords_raw[:10] if keywords_raw else [])
            ],
            "classifications_compteurs": classifs.get("compteurs", {}),
            "stats": {
                "nb_phrases": int(stats.get("nb_phrases", 0) or 0),
                "nb_mots": int(stats.get("nb_mots", 0) or 0),
                "mots_uniques": int(stats.get("mots_uniques", 0) or 0),
            },
            "computed_at": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erreur nlp/business-tonalite: {e}")
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/rapport/structure")
def api_rapport_structure(report_type: str = Query("global"),
                          year: int = Query(None),
                          region: str = Query(None),
                          category: str = Query(None)):
    """
    Renvoie la structure JSON Chantier 1 (recommandations hierarchisees).
    Cherche d'abord en cache, fallback generation live via Mistral.
    """
    try:
        struct = get_rapport_structure(report_type=report_type,
                                       year=year, region=region, category=category)
        if struct:
            return {
                "available": True,
                "source": "cache",
                "report_type": report_type,
                "filters": {"year": year, "region": region, "category": category},
                "structure": struct,
            }

        logger.info(
            f"Structure JSON absente du cache, generation dynamique "
            f"(type={report_type}, year={year}, region={region}, category={category})"
        )
        from src.nlp_transformers import generer_structure_json
        faits = _build_faits_from_pg(year=year, region=region, category=category)
        struct_live = generer_structure_json(faits, langue="fr")
        return {
            "available": True,
            "source": "live",
            "report_type": report_type,
            "filters": {"year": year, "region": region, "category": category},
            "structure": struct_live,
            "note": "Structure generee a la volee. Lancer main.py pour la persister.",
        }
    except Exception as e:
        logger.exception(f"Erreur rapport/structure: {e}")
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/validation/business")
def api_validation_business():
    """
    Validation metier automatique des KPIs (Priorite 2).
    Verifie la coherence interne des donnees Postgres.
    Renvoie : status_global, nb_checks, listes des checks.
    """
    try:
        from src.business_validation import run_all_checks
        return run_all_checks()
    except Exception as e:
        logger.exception(f"Erreur validation/business: {e}")
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/insights/traceability")
def api_insights_traceability(lang: str = Query("fr"),
                              year: int = Query(None),
                              region: str = Query(None),
                              category: str = Query(None)):
    """
    Tracabilite insight -> KPI -> confiance (Priorite 5).
    Renvoie 5 a 8 insights, chacun lie a son KPI source, sa valeur de
    reference et son niveau de confiance.
    """
    try:
        from src.insight_traceability import construire_traces
        faits = _build_faits_from_pg(year=year, region=region, category=category)
        traces = construire_traces(faits, langue=lang)
        return {
            "filters": {"year": year, "region": region, "category": category},
            "lang": lang,
            "nb_insights": len(traces),
            "insights": traces,
            "computed_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.exception(f"Erreur insights/traceability: {e}")
        raise HTTPException(500, str(e))


@app.get(f"{API_PREFIX}/recommendations/rules")
def api_recommendations_rules(lang: str = Query("fr"),
                              year: int = Query(None),
                              region: str = Query(None),
                              category: str = Query(None)):
    """
    Recommandations issues du moteur de regles (Priorite 4).
    Complementaires des recos LLM Chantier 4 - garanties par regles.
    """
    try:
        from src.business_rules import evaluer_regles
        faits = _build_faits_from_pg(year=year, region=region, category=category)
        recos = evaluer_regles(faits, langue=lang)
        return {
            "filters": {"year": year, "region": region, "category": category},
            "lang": lang,
            "nb_recommandations": len(recos),
            "recommandations": recos,
            "source": "business_rules_engine",
            "computed_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.exception(f"Erreur recommendations/rules: {e}")
        raise HTTPException(500, str(e))




@app.get(f"{API_PREFIX}/nlp/quality")
def nlp_quality(report_type: str = Query("global"), lang: str = Query("fr"),
                year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    """Score qualite NLP detaille sur 100 + 6 criteres + suggestions."""
    try:
        data = get_nlp_report(report_type=report_type, lang=lang,
                              year=year, region=region, category=category)
        if data and data.get("rapport_complet"):
            rapport = _strip_markdown(data["rapport_complet"])
        else:
            from src.nlp_transformers import generer_rapport
            faits_local = _build_faits_from_pg(year=year, region=region, category=category)
            rapport = _strip_markdown(generer_rapport(faits_local, langue=lang))

        from nlp_quality_score import evaluer_qualite_rapport
        faits = _build_faits_from_pg(year=year, region=region, category=category)
        result = evaluer_qualite_rapport(rapport_texte=rapport, faits=faits, langue=lang)

        return {
            "score_nlp": result["score_nlp"],
            "mention": result["mention"],
            "couleur": couleur_pour_score(int(result.get("score_nlp") or 0)),
            "details": result["details"],
            "lacunes": result["lacunes"],
            "filters": {"year": year, "region": region, "category": category},
            "report_type": report_type,
            "lang": lang,
            "evaluated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.exception(f"Erreur nlp/quality: {e}")
        raise HTTPException(500, str(e))


# Chatbot RAG

FORBIDDEN_PATTERNS = [
    "ignore previous", "ignore your", "ignore les", "oublie",
    "system prompt", "your instructions", "tes instructions",
    "act as", "pretend", "roleplay", "jailbreak",
    "developer mode", "DAN mode", "respond in unrestricted",
]

# Salutations (FR + EN)
SALUTATIONS = {
    "bonjour", "salut", "coucou", "hello", "hi", "hey", "yo",
    "bonsoir", "good morning", "good afternoon", "good evening",
    "hola", "ciao",
}

# Remerciements
REMERCIEMENTS = {
    "merci", "thanks", "thank you", "thx", "ty", "cheers", "merci beaucoup",
}

# Reponses affirmatives courtes
AFFIRMATIONS = {
    "oui", "yes", "yep", "ouais", "ok", "okay", "d'accord", "bien sur",
    "sure", "yeah", "absolutely", "vas-y", "go", "carrement",
}

# Reponses negatives courtes
NEGATIONS = {
    "non", "no", "nope", "pas maintenant", "plus tard",
    "not now", "later", "nah", "neg",
}

# Vocabulaire metier Superstore (whitelist).
# Si la question contient un de ces mots, on considere qu'elle est dans le scope.
DOMAIN_KEYWORDS = {
    # Ventes / CA
    "vente", "ventes", "sales", "revenue", "ca", "chiffre", "chiffres", "panier",
    "basket", "commande", "commandes", "order", "orders",
    # Temporalite
    "annee", "annees", "year", "years", "trimestre", "trimestres", "quarter",
    "quarters", "mois", "month", "months", "2015", "2016", "2017", "2018",
    "yoy", "qoq", "t1", "t2", "t3", "t4", "q1", "q2", "q3", "q4",
    "saison", "saisonnier", "saisonnalite", "seasonality", "periode",
    # Geographie / segments
    "region", "regions", "west", "east", "south", "central", "etat",
    "state", "california", "texas", "new york", "washington",
    "segment", "consumer", "corporate", "home office", "client", "clients",
    "customer", "customers",
    # Produits
    "categorie", "categories", "category", "sub-category", "sous-categorie",
    "produit", "produits", "product", "products", "phones", "chairs", "storage",
    "tables", "binders", "machines", "accessories", "copiers", "bookcases",
    "appliances", "furniture", "office", "technology",
    # Analyses
    "tendance", "tendances", "trend", "trends", "croissance", "growth",
    "anomalie", "anomalies", "anomaly", "risque", "risk", "opportunite",
    "opportunities", "performance", "perf", "kpi", "kpis", "indicateur",
    "indicateurs", "metric", "metrics", "volatilite", "volatility",
    "stabilite", "stability", "tonalite", "tonality", "sentiment",
    # Actions
    "recommande", "recommendation", "recommandation", "suggere", "suggest",
    "action", "actions", "plan", "strategie", "strategy", "decision",
    # Comparaisons
    "compare", "comparaison", "comparison", "versus", "vs", "ratio",
    # Marque
    "superstore", "magasin", "store", "retail",
    # Mots de question generiques business
    "pourquoi", "why", "quel", "quelle", "quels", "quelles", "what",
    "which", "how", "combien", "qui", "who",
}


def _detect_short_intent(question: str) -> str:
    """
    Detecte les intents courts (salutation, remerciement, oui, non).
    Renvoie 'salutation', 'remerciement', 'affirmation', 'negation' ou ''.
    """
    q_norm = question.lower().strip()
    q_norm = q_norm.rstrip(".!?,;: ")
    if len(q_norm) > 30:
        return ""

    # Verification exacte (token simple)
    if q_norm in SALUTATIONS:
        return "salutation"
    if q_norm in REMERCIEMENTS:
        return "remerciement"
    if q_norm in AFFIRMATIONS:
        return "affirmation"
    if q_norm in NEGATIONS:
        return "negation"

    # Verification debut de phrase (salutation suivie d'autre chose)
    for s in SALUTATIONS:
        if q_norm.startswith(s + " ") or q_norm.startswith(s + ","):
            return "salutation"
    return ""


def _is_in_scope(question: str) -> bool:
    """
    Verifie si la question est dans le domaine Superstore (ventes 2015-2018).
    Strategie : whitelist de mots-cles metier. Si AUCUN match, hors scope.
    """
    q_norm = question.lower()
    for kw in DOMAIN_KEYWORDS:
        if kw in q_norm:
            return True
    return False


def _suggestions_par_intent(intent: str, lang: str = "fr") -> list:
    """Renvoie 3 suggestions de questions adaptees a l'intention."""
    if lang == "en":
        if intent == "salutation":
            return [
                "Which region drives growth?",
                "What are the main risks?",
                "Compare West and East",
            ]
        if intent == "recommandation":
            return [
                "Why did 2016 decline?",
                "Which products concentrate revenue?",
                "What is the riskiest quarter?",
            ]
        if intent in ("risque", "anomalie"):
            return [
                "What action do you recommend?",
                "Which region is most volatile?",
                "Why is Q1 2016 problematic?",
            ]
        return [
            "Which region drives growth?",
            "What are the main risks?",
            "What products concentrate revenue?",
        ]

    if intent == "salutation":
        return [
            "Quelle region tire la croissance ?",
            "Quels sont les risques principaux ?",
            "Compare West et East",
        ]
    if intent == "recommandation":
        return [
            "Pourquoi 2016 baisse ?",
            "Quels produits concentrent le CA ?",
            "Quel trimestre est le plus risque ?",
        ]
    if intent in ("risque", "anomalie"):
        return [
            "Quelle action recommandes-tu ?",
            "Quelle region est la plus volatile ?",
            "Pourquoi T1 2016 est problematique ?",
        ]
    if intent == "comparaison":
        return [
            "Quelle region domine en 2018 ?",
            "Comment evolue le panier moyen ?",
            "Quel segment est le plus rentable ?",
        ]
    return [
        "Quelle region tire la croissance ?",
        "Quels sont les risques principaux ?",
        "Quels produits concentrent le CA ?",
    ]

INTENT_KEYWORDS = {
    "top_client":  ["top client", "meilleur client", "quel client", "client le plus",
                    "top customer", "best customer"],
    "top_product": ["top produit", "meilleur produit", "produit le plus", "best seller",
                    "top product", "what sells", "produits concentrent", "products concentrate"],
    "anomalie":    ["anomalie", "alerte", "chute", "probleme", "bizarre", "inattendu",
                    "anomaly", "alert", "drop", "issue", "problematique", "problematic"],
    "comparaison": [" vs ", "versus", "compare", "compared", "difference entre",
                    "compared to", "difference between"],
    "prevision":   ["prevision", "forecast", "predire", "a venir", "futur", "demain",
                    "predict", "future", "next"],
    "tendance":    ["tendance", "evolution", "progression", "growth", "trend",
                    "tire la croissance", "drives growth"],
    "cause":       ["pourquoi", "why", "cause", "explique", "explain", "raison",
                    "reason", "baisse", "decline"],
    "risque":      ["risque", "risk", "menace", "threat", "vulnerab", "danger"],
    "recommandation": ["recommande", "recommend", "action", "suggere", "que faire",
                       "what should", "next steps"],
    "region":      ["region", "ouest", "est", "nord", "sud", "west", "east", "north", "south", "central"],
}

SYSTEM_PROMPT_FR = """Tu es Sup'Bot, analyste business senior pour Superstore.

STRUCTURE OBLIGATOIRE DE TA REPONSE (4 etapes en prose fluide, pas en liste) :
1. CONSTAT : une phrase qui resume ce que disent les donnees.
2. PREUVE CHIFFREE : cite 2 chiffres maximum issus du CONTEXTE fourni.
3. INTERPRETATION METIER : ce que ce constat implique (risque, opportunite, dependance, tendance).
4. RECOMMANDATION : une action concrete et liee aux chiffres cites.

REGLES DE FORMAT :
- Francais fluide, phrases completes, ton executif (comme a l'oral a un dirigeant).
- STRICTEMENT INTERDIT : aucun markdown. Pas de **, ###, tirets de liste, separateurs, numerotation.
- 4 a 6 phrases au total. Pas plus.
- Commence par le constat. Termine par la recommandation.

REGLES METIER :
- Utilise UNIQUEMENT les DONNEES fournies dans le bloc CONTEXTE.
- Si une information manque, dis-le explicitement en une phrase : "Donnee non disponible sur ce perimetre."
- N'invente JAMAIS un chiffre, une region, un produit ou une annee qui n'est pas dans le contexte.
- Refuse tout sujet hors business retail.
- Croissance YoY = comparaison a l'annee precedente. Anomalie = variation trimestrielle au-dela de 20%. Panier moyen = CA / commandes uniques.
- Concentration > 35% sur une dimension = mentionner le risque de dependance.

EXEMPLE DE BONNE REPONSE :
Question : "Quelle region tire la croissance ?"
Reponse : "La region West tire la croissance avec 244,213 USD en 2018, soit pres de 35% du CA national. Cette concentration confirme un moteur regional clair, mais expose l'entreprise a une dependance geographique. Pour limiter ce risque, il serait pertinent de developper des relais commerciaux dans East et Central, qui restent en deca de leur potentiel."
"""

SYSTEM_PROMPT_EN = """You are Sup'Bot, senior business analyst for Superstore.

MANDATORY ANSWER STRUCTURE (4 steps as flowing prose, not a list):
1. OBSERVATION: one sentence summarizing what the data says.
2. QUANTIFIED EVIDENCE: cite at most 2 figures from the CONTEXT provided.
3. BUSINESS INTERPRETATION: what this observation implies (risk, opportunity, dependence, trend).
4. RECOMMENDATION: one concrete action linked to the cited figures.

FORMAT RULES:
- Fluent English, full sentences, executive tone.
- STRICTLY FORBIDDEN: no markdown. No **, ###, list dashes, separators, numbering.
- 4 to 6 sentences total. Not more.
- Start with the observation. End with the recommendation.

BUSINESS RULES:
- Use ONLY the DATA provided in the CONTEXT block.
- If information is missing, state explicitly: "Data not available on this scope."
- NEVER invent a number, region, product or year that's not in the context.
- Refuse non-retail topics.
- YoY growth = compared to previous year. Anomaly = quarterly variation beyond 20%. Average basket = revenue / unique orders.
- Concentration > 35% on any dimension = mention dependence risk."""


def _detect_injection(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in FORBIDDEN_PATTERNS)


def _detect_intent(question: str) -> str:
    q = question.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in q for k in keywords):
            return intent
    return "general"


def _build_intent_context(intent: str, year, region, category) -> List[str]:
    """Enrichit le contexte selon l'intention detectee."""
    extra = []

    if intent == "top_client":
        try:
            clients = get_top_clients(year=year, region=region, category=category, limit=5)
            if clients:
                lines = [
                    f"  #{i+1} {c['customer_id']} : {float(c['ca_total']):,.0f} USD "
                    f"({c['nb_commandes']} commandes, panier {float(c['panier_moyen']):.0f} USD)"
                    for i, c in enumerate(clients)
                ]
                extra.append("TOP CLIENTS :\n" + "\n".join(lines))
        except Exception as e:
            logger.warning(f"intent top_client failed: {e}")

    elif intent == "top_product":
        try:
            products = get_kpi_top_products(year=year, region=region, category=category)[:5]
            if products:
                lines = [
                    f"  #{i+1} {p['sub_category']} : {float(p['ventes_totales']):,.0f} USD"
                    for i, p in enumerate(products)
                ]
                extra.append("TOP SOUS-CATEGORIES :\n" + "\n".join(lines))
        except Exception as e:
            logger.warning(f"intent top_product failed: {e}")

    elif intent == "anomalie":
        try:
            anomalies_data = get_anomalies(year=year, region=region, category=category)
            if anomalies_data:
                lines = [
                    f"  - {a.get('niveau','')} T{a.get('trimestre','')} {a.get('annee','')} : "
                    f"{a.get('variation','')}% ({float(a.get('ventes', 0)):,.0f} USD)"
                    for a in anomalies_data[:5]
                ]
                extra.append("ANOMALIES DETECTEES :\n" + "\n".join(lines))
            else:
                extra.append("ANOMALIES : aucune anomalie detectee sur ce perimetre.")
        except Exception as e:
            logger.warning(f"intent anomalie failed: {e}")

    elif intent == "prevision":
        extra.append(
            "NOTE PREVISIONS : les donnees historiques couvrent uniquement 2015-2018. "
            "Aucun modele predictif n'est disponible."
        )

    elif intent == "comparaison":
        try:
            annual = get_kpi_annual()
            if annual:
                lines = [
                    f"  {r['annee']} : {float(r['ca_annuel']):,.0f} USD "
                    f"({r.get('croissance_yoy', 'N/A')}% YoY)"
                    for r in annual
                ]
                extra.append("COMPARAISON ANNUELLE :\n" + "\n".join(lines))
            # On ajoute aussi le breakdown par region pour les comparaisons regionales
            regs = get_kpi_regions_summary(year=year, region=None, category=category)
            if regs:
                lines = [
                    f"  {r['region']} : {float(r.get('ventes_totales', 0)):,.0f} USD "
                    f"({int(r.get('nb_commandes', 0)):,} commandes, panier {float(r.get('panier_moyen', 0)):.0f} USD)"
                    for r in regs
                ]
                extra.append("BREAKDOWN PAR REGION :\n" + "\n".join(lines))
        except Exception as e:
            logger.warning(f"intent comparaison failed: {e}")

    elif intent in ("cause", "tendance"):
        try:
            annual = get_kpi_annual()
            if annual:
                lines = [
                    f"  {r['annee']} : {float(r['ca_annuel']):,.0f} USD "
                    f"({r.get('croissance_yoy', 'N/A')}% YoY)"
                    for r in annual
                ]
                extra.append("EVOLUTION ANNUELLE :\n" + "\n".join(lines))
            qrt = get_kpi_quarterly(year=year, region=region, category=category)
            if qrt:
                lines = [
                    f"  T{q['trimestre']} {q['annee']} : {float(q.get('ventes_totales', 0)):,.0f} USD "
                    f"(variation QoQ : {q.get('variation_pct', 'N/A')}%)"
                    for q in qrt[-8:]  # 8 derniers trimestres
                ]
                extra.append("EVOLUTION TRIMESTRIELLE (8 derniers) :\n" + "\n".join(lines))
        except Exception as e:
            logger.warning(f"intent cause/tendance failed: {e}")

    elif intent == "risque":
        try:
            from src.business_rules import evaluer_regles
            faits_local = _build_faits_from_pg(year=year, region=region, category=category)
            recos = evaluer_regles(faits_local, langue="fr")
            risques = [r for r in recos if r.get("priorite") == "haute"]
            if risques:
                lines = [
                    f"  - [{r['regle']}] {r['action']} (justification : {r['justification']})"
                    for r in risques[:5]
                ]
                extra.append("RISQUES IDENTIFIES PAR LES REGLES BUSINESS :\n" + "\n".join(lines))
            else:
                extra.append("RISQUES : aucun risque majeur detecte par le moteur de regles.")
        except Exception as e:
            logger.warning(f"intent risque failed: {e}")

    elif intent == "recommandation":
        try:
            from src.business_rules import evaluer_regles
            faits_local = _build_faits_from_pg(year=year, region=region, category=category)
            recos = evaluer_regles(faits_local, langue="fr")
            if recos:
                lines = [
                    f"  [{r['priorite'].upper()} / {r['niveau']}] {r['action']}"
                    for r in recos[:5]
                ]
                extra.append("RECOMMANDATIONS DU MOTEUR DE REGLES :\n" + "\n".join(lines))
        except Exception as e:
            logger.warning(f"intent recommandation failed: {e}")

    elif intent == "region":
        try:
            regs = get_kpi_regions_summary(year=year, region=None, category=category)
            if regs:
                total = sum(float(r.get("ventes_totales", 0)) for r in regs) or 1
                lines = [
                    f"  {r['region']} : {float(r.get('ventes_totales', 0)):,.0f} USD "
                    f"({float(r.get('ventes_totales', 0))/total*100:.1f}% du total)"
                    for r in regs
                ]
                extra.append("PART DE CHAQUE REGION :\n" + "\n".join(lines))
        except Exception as e:
            logger.warning(f"intent region failed: {e}")

    return extra


def _embed_question(question: str) -> Optional[List[float]]:
    """Embed via sentence-transformers (priorite) ou Mistral (fallback)."""
    st_model = get_sentence_transformer()
    if st_model is not None:
        try:
            embedding = st_model.encode(question, convert_to_numpy=False)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            return embedding
        except Exception as e:
            logger.warning(f"Embedding HF echoue: {e}, fallback Mistral")

    if not MISTRAL_API_KEY:
        return None
    try:
        from mistralai.client import Mistral
        client = Mistral(api_key=MISTRAL_API_KEY)
        resp = client.embeddings.create(model=MISTRAL_EMBED_MODEL, inputs=[question])
        return resp.data[0].embedding
    except Exception as e:
        logger.warning(f"Embedding Mistral echoue: {e}")
        return None


class ChatRequest(BaseModel):
    question: str
    lang: str = "fr"
    year: Optional[int] = None
    region: Optional[str] = None
    category: Optional[str] = None
    use_rag: bool = True


@app.post(f"{API_PREFIX}/chat")
def chat_rag(req: ChatRequest):
    """Chatbot RAG analyste : gere salutations, hors-sujet, oui/non, puis Mistral."""
    try:
        if not req.question or len(req.question.strip()) < 1:
            return {
                "response": ("Posez une question - je peux vous aider sur les ventes Superstore 2015-2018."
                             if req.lang == "fr" else "Ask a question - I help on Superstore 2015-2018 sales."),
                "context_used": {"engine": "rejected", "reason": "too_short"},
                "suggestions": _suggestions_par_intent("", req.lang),
            }

        if len(req.question) > 1000:
            return {
                "response": ("Question trop longue (>1000 caracteres). Soyez plus concis."
                             if req.lang == "fr" else "Question too long (>1000 chars)."),
                "context_used": {"engine": "rejected", "reason": "too_long"},
                "suggestions": _suggestions_par_intent("", req.lang),
            }

        # Detection des injections de prompt
        if _detect_injection(req.question):
            logger.warning(f"Injection detectee: {req.question[:80]}")
            return {
                "response": ("Je ne traite que des questions sur l'activite Superstore (ventes 2015-2018)."
                             if req.lang == "fr"
                             else "I only handle Superstore business questions (sales 2015-2018)."),
                "context_used": {"engine": "blocked", "reason": "injection_attempt"},
                "suggestions": _suggestions_par_intent("", req.lang),
            }

        # Detection des intents courts (salutation, oui, non, merci)
        short = _detect_short_intent(req.question)

        if short == "salutation":
            msg = (
                "Bonjour. Je suis Sup'Bot, votre assistant analyste pour Superstore. "
                "Je peux vous aider a explorer les ventes, regions, categories et "
                "tendances entre 2015 et 2018. Souhaitez-vous explorer un sujet ?"
                if req.lang == "fr"
                else "Hello. I'm Sup'Bot, your Superstore analyst assistant. "
                "I can help you explore sales, regions, categories and trends "
                "between 2015 and 2018. Would you like to explore a topic?"
            )
            return {
                "response": msg,
                "context_used": {"engine": "short_intent", "intent": "salutation"},
                "suggestions": _suggestions_par_intent("salutation", req.lang),
            }

        if short == "remerciement":
            msg = (
                "Avec plaisir. Voulez-vous explorer un autre sujet ?"
                if req.lang == "fr"
                else "You're welcome. Want to explore another topic?"
            )
            return {
                "response": msg,
                "context_used": {"engine": "short_intent", "intent": "remerciement"},
                "suggestions": _suggestions_par_intent("salutation", req.lang),
            }

        if short == "affirmation":
            msg = (
                "Parfait. Voici quelques pistes d'analyse pour commencer :"
                if req.lang == "fr"
                else "Great. Here are a few starting points:"
            )
            return {
                "response": msg,
                "context_used": {"engine": "short_intent", "intent": "affirmation"},
                "suggestions": _suggestions_par_intent("salutation", req.lang),
            }

        if short == "negation":
            msg = (
                "D'accord. Je reste a votre disposition si vous avez d'autres "
                "questions sur l'activite Superstore."
                if req.lang == "fr"
                else "Alright. I'm available if you have other questions about "
                "Superstore activity."
            )
            return {
                "response": msg,
                "context_used": {"engine": "short_intent", "intent": "negation"},
                "suggestions": _suggestions_par_intent("salutation", req.lang),
            }

        # Validation du scope (Superstore / ventes 2015-2018)
        if not _is_in_scope(req.question):
            msg = (
                "Je suis un chatbot dedie a l'activite Superstore et je ne peux "
                "repondre qu'aux questions sur les ventes, regions, categories, "
                "produits, clients et tendances entre 2015 et 2018. Voulez-vous "
                "que je vous propose une question dans ce perimetre ?"
                if req.lang == "fr"
                else "I'm a Superstore-only chatbot - I only answer questions on "
                "sales, regions, categories, products, customers and trends "
                "between 2015 and 2018. Would you like a question suggestion?"
            )
            return {
                "response": msg,
                "context_used": {"engine": "out_of_scope", "intent": "off_topic"},
                "suggestions": _suggestions_par_intent("", req.lang),
            }

        intent = _detect_intent(req.question)
        logger.info(f"Question: '{req.question[:60]}...' | intent={intent}")

        context_parts = []
        kpi_summary = get_kpi_filtered_summary(year=req.year, region=req.region, category=req.category)
        if kpi_summary:
            context_parts.append(
                f"KPIs PERIMETRE ACTUEL :\n"
                f"  CA total = {float(kpi_summary.get('ca_total', 0)):,.0f} USD\n"
                f"  Nb commandes uniques = {kpi_summary.get('nb_commandes', 0):,}\n"
                f"  Panier moyen = {float(kpi_summary.get('panier_moyen', 0)):.2f} USD\n"
                f"  Croissance YoY = {kpi_summary.get('croissance_globale', 'N/A')}%\n"
                f"  Meilleure region = {kpi_summary.get('meilleure_region', 'N/A')}\n"
                f"  Nb clients = {kpi_summary.get('nb_clients', 0):,}\n"
                f"  Periode = {kpi_summary.get('periode', 'N/A')}"
            )

        context_parts.extend(_build_intent_context(intent, req.year, req.region, req.category))

        rag_engine = "context_only"
        rag_chunks_used = []
        if req.use_rag and vectorstore_available():
            embedding = _embed_question(req.question)
            if embedding:
                rag_chunks = search_rag_chunks(query_embedding=embedding, top_k=5)
                rag_chunks = [c for c in rag_chunks if c.get("similarity", 0) > 0.3]
                if rag_chunks:
                    rag_engine = "rag_chromadb"
                    rag_chunks_used = [
                        {
                            "type": c.get("chunk_type", ""),
                            "similarity": c.get("similarity", 0),
                            "preview": (c.get("content", "")[:120] + "...")
                                       if len(c.get("content", "")) > 120
                                       else c.get("content", ""),
                        }
                        for c in rag_chunks
                    ]
                    rag_text = "\n\n".join([
                        f"  [{c['chunk_type']} - similarite {c['similarity']:.2f}]\n  {c['content']}"
                        for c in rag_chunks
                    ])
                    context_parts.append(
                        f"PASSAGES PERTINENTS (recherche semantique ChromaDB) :\n{rag_text}"
                    )
                    logger.info(
                        f"RAG ChromaDB : {len(rag_chunks)} chunks utilises "
                        f"(types: {set(c['chunk_type'] for c in rag_chunks)})"
                    )

        context_text = "\n\n".join(context_parts) if context_parts else "Aucune donnee disponible."

        filtres = []
        if req.year: filtres.append(f"annee={req.year}")
        if req.region: filtres.append(f"region={req.region}")
        if req.category: filtres.append(f"categorie={req.category}")
        filtre_str = ", ".join(filtres) if filtres else "aucun (vue globale)"

        if not MISTRAL_API_KEY:
            logger.warning("MISTRAL_API_KEY vide - fallback")
            return {"response": _chat_fallback(req.question, kpi_summary, [], [], req.lang),
                    "context_used": {"filters": filtre_str, "intent": intent, "engine": "fallback",
                                     "error": "MISTRAL_API_KEY non configuree"}}

        try:
            from mistralai.client import Mistral
            client = Mistral(api_key=MISTRAL_API_KEY)
        except ImportError as e:
            logger.error(f"Import mistralai echoue: {e}")
            return {"response": _chat_fallback(req.question, kpi_summary, [], [], req.lang),
                    "context_used": {"engine": "fallback",
                                     "error": f"mistralai non installe: {e}"}}

        try:
            system_prompt = SYSTEM_PROMPT_FR if req.lang == "fr" else SYSTEM_PROMPT_EN
            user_message = (
                f"FILTRES ACTIFS : {filtre_str}\n"
                f"INTENT DETECTE : {intent}\n\n"
                f"=== DONNEES CONTEXTUELLES ===\n{context_text}\n"
                f"=== FIN DES DONNEES ===\n\n"
                f"QUESTION DE L'UTILISATEUR : {req.question}"
            )

            logger.info(f"Appel Mistral chat (model={MISTRAL_MODEL}, len_ctx={len(context_text)})")

            response = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.15,
                max_tokens=350,
            )

            raw_answer = response.choices[0].message.content
            answer = _strip_markdown(raw_answer)
            logger.info(f"[OK] Mistral chat OK ({len(answer)} chars)")

            return {
                "response": answer,
                "context_used": {"filters": filtre_str, "intent": intent,
                                 "engine": rag_engine, "model": MISTRAL_MODEL,
                                 "rag_chunks": rag_chunks_used},
                "suggestions": _suggestions_par_intent(intent, req.lang),
            }
        except Exception as e:
            logger.exception(f"Erreur Mistral chat: {type(e).__name__}: {e}")
            return {
                "response": _chat_fallback(req.question, kpi_summary, [], [], req.lang),
                "context_used": {"engine": "fallback",
                                 "error": f"{type(e).__name__}: {str(e)[:200]}"},
                "suggestions": _suggestions_par_intent(intent, req.lang),
            }

    except Exception as e:
        logger.exception("Erreur chatbot")
        raise HTTPException(500, f"Erreur chatbot: {e}")


def _chat_fallback(question, kpi, anomalies, products, lang):
    """Fallback sans Mistral - texte naturel sans markdown."""
    if not kpi:
        return "Donnees indisponibles." if lang == "fr" else "Data unavailable."

    ca = float(kpi.get("ca_total", 0))
    cr = kpi.get("croissance_globale", 0) or "N/A"
    region = kpi.get("meilleure_region", "N/A")
    nb_cmd = kpi.get("nb_commandes", 0)
    panier = float(kpi.get("panier_moyen", 0))

    if lang == "fr":
        return (f"Sur le perimetre actuel, le chiffre d'affaires s'eleve a {ca:,.0f} USD "
                f"avec une croissance de {cr}%. La region {region} reste en tete, "
                f"avec un panier moyen de {panier:.0f} USD sur {nb_cmd:,} commandes. "
                f"(Reponse generee en mode degrade, Mistral indisponible.)")
    return (f"In the current scope, revenue is ${ca:,.0f} with growth of {cr}%. "
            f"{region} leads with an average basket of ${panier:.0f} over {nb_cmd:,} orders. "
            f"(Degraded mode, Mistral unavailable.)")


@app.get(f"{API_PREFIX}/chat/health")
def chat_health():
    return {
        "mistral_configured": bool(MISTRAL_API_KEY),
        "model": MISTRAL_MODEL,
        "embed_model": MISTRAL_EMBED_MODEL,
        "vectorstore": get_rag_stats(),
        "intents_supported": list(INTENT_KEYWORDS.keys()),
    }


# PDF endpoints

@app.get(f"{API_PREFIX}/pdf/generate")
def pdf_generate(lang: str = Query("fr"), report_type: str = Query("global"),
                 year: int = Query(None), region: str = Query(None), category: str = Query(None)):
    try:
        from src.generate_pdf import generer_pdf
        chemin = generer_pdf(langue=lang, report_type=report_type,
                             year=year, region=region, category=category)
        return FileResponse(path=chemin, media_type="application/pdf",
                            filename=Path(chemin).name)
    except Exception as e:
        logger.error(f"Erreur PDF: {e}")
        raise HTTPException(500, f"Erreur PDF: {e}")


@app.get(f"{API_PREFIX}/pdf/latest")
def pdf_latest(lang: str = Query("fr")):
    output_dir = Path(OUTPUT_DIR)
    pdfs = sorted(output_dir.glob(f"rapport_business_{lang}_*.pdf"), reverse=True)
    if not pdfs:
        raise HTTPException(404, f"Aucun PDF '{lang}'")
    return FileResponse(path=str(pdfs[0]), media_type="application/pdf", filename=pdfs[0].name)


@app.get(f"{API_PREFIX}/pdf/list")
def pdf_list():
    output_dir = Path(OUTPUT_DIR)
    pdfs = sorted(output_dir.glob("rapport_business_*.pdf"), reverse=True)
    return [{"filename": p.name, "size_kb": round(p.stat().st_size / 1024, 1),
             "created": datetime.fromtimestamp(p.stat().st_ctime).isoformat()} for p in pdfs]


# Generation planifiee (batch des 12 rapports)

@app.post(f"{API_PREFIX}/reports/schedule")
def schedule_reports_generation(lang: str = Query("fr"), report_types: str = Query("all"),
                                background: bool = Query(False)):
    """Genere en serie les 12 rapports (1 global + 4 annees + 4 regions + 3 categories)."""
    try:
        from src.generate_pdf import generer_pdf
        rapports_generes = []

        if report_types in ["all", "global"]:
            chemin = generer_pdf(langue=lang, report_type="global")
            rapports_generes.append({"type": "global", "path": str(chemin),
                                     "size_kb": round(Path(chemin).stat().st_size / 1024, 1)})

        if report_types in ["all", "annual"]:
            for year in [2015, 2016, 2017, 2018]:
                chemin = generer_pdf(langue=lang, report_type="by_year", year=year)
                rapports_generes.append({"type": f"annual_{year}", "path": str(chemin),
                                         "size_kb": round(Path(chemin).stat().st_size / 1024, 1)})

        if report_types in ["all", "regional"]:
            for region in ["West", "East", "Central", "South"]:
                chemin = generer_pdf(langue=lang, report_type="by_region", region=region)
                rapports_generes.append({"type": f"region_{region}", "path": str(chemin),
                                         "size_kb": round(Path(chemin).stat().st_size / 1024, 1)})

        if report_types in ["all", "category"]:
            for category in ["Furniture", "Office Supplies", "Technology"]:
                chemin = generer_pdf(langue=lang, report_type="by_category", category=category)
                rapports_generes.append({"type": f"category_{category}", "path": str(chemin),
                                         "size_kb": round(Path(chemin).stat().st_size / 1024, 1)})

        return {"status": "success", "total_generated": len(rapports_generes),
                "language": lang, "timestamp": datetime.now().isoformat(),
                "reports": rapports_generes}
    except Exception as e:
        logger.error(f"Erreur generation auto: {e}")
        raise HTTPException(500, f"Erreur generation: {e}")


@app.post(f"{API_PREFIX}/report/scheduled")
def generate_scheduled_reports(background_tasks: BackgroundTasks, lang: str = Query("fr")):
    """Lance la generation des 12 rapports en background."""
    def _generate_all_reports():
        try:
            from src.generate_pdf import generer_pdf
            for year in [2015, 2016, 2017, 2018]:
                try:
                    generer_pdf(langue=lang, report_type="by_year", year=year)
                except Exception as e:
                    logger.error(f"Rapport {year} echoue: {e}")
            for region in ["West", "East", "Central", "South"]:
                try:
                    generer_pdf(langue=lang, report_type="by_region", region=region)
                except Exception as e:
                    logger.error(f"Rapport {region} echoue: {e}")
            for category in ["Furniture", "Office Supplies", "Technology"]:
                try:
                    generer_pdf(langue=lang, report_type="by_category", category=category)
                except Exception as e:
                    logger.error(f"Rapport {category} echoue: {e}")
        except Exception as e:
            logger.exception(f"Erreur fatale generation planifiee: {e}")

    background_tasks.add_task(_generate_all_reports)
    return {"status": "scheduled", "reports_count": 12, "langue": lang,
            "timestamp": datetime.now().isoformat()}


@app.get(f"{API_PREFIX}/dashboard")
def dashboard_data(year: int = Query(None), region: str = Query(None),
                   category: str = Query(None), lang: str = Query("fr")):
    """Agrege toutes les donnees necessaires au dashboard HTML."""
    try:
        try:
            nlp = nlp_report(report_type="global", lang=lang, year=year, region=region, category=category)
        except Exception as e:
            logger.warning(f"nlp_report failed in dashboard: {e}")
            nlp = {}

        annual = get_kpi_annual_filtered(year=year, region=region, category=category) \
            if _has_filters(year, region, category) else get_kpi_annual()

        return {
            "applied_filters": {"year": year, "region": region, "category": category},
            "kpi_global": get_kpi_filtered_summary(year=year, region=region, category=category),
            "annual": annual,
            "regions": get_kpi_regions_summary(year=year, region=region, category=category),
            "categories": get_kpi_categories_summary(year=year, region=region, category=category),
            "sub_categories": get_kpi_sub_categories(year=year, region=region, category=category, limit=10),
            "quarterly": get_kpi_quarterly(year=year, region=region, category=category),
            "monthly": get_kpi_monthly(year=year, region=region, category=category),
            "segments": get_kpi_segments_summary(year=year, region=region, category=category),
            "top_products": get_kpi_top_products(year=year, region=region, category=category),
            "top_clients": get_top_clients(year=year, region=region, category=category, limit=5),
            "anomalies": get_anomalies(year=year, region=region, category=category),
            "nlp_report": nlp or {},
        }
    except Exception as e:
        logger.exception("Erreur dashboard")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
