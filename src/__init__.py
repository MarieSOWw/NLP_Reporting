"""
NLP Reporting — Module Source
==============================
Pipeline de génération automatique de rapports business en langage naturel.

Architecture:
    DATA → Spark (analytics) → PostgreSQL → FastAPI (api) → Frontend / PDF

Modules:
    - load_data: Chargement et nettoyage des données avec Spark
    - analytics: Calcul des KPIs business (source unique de calcul)
    - nlp_nltk: Extraction des faits et analyse NLTK
    - nlp_transformers: Génération de rapport avec Mistral API
    - save_to_postgres: Export vers PostgreSQL (source unique de vérité)
    - db: Lecture centralisée depuis PostgreSQL (pour API et PDF)
    - generate_pdf: Export PDF avec ReportLab
    - api: Serveur FastAPI (couche REST + chatbot RAG)

Usage:
    # Pipeline complet
    from src.load_data import create_spark_session, load_data
    from src.analytics import compute_kpis
    from src.nlp_nltk import extraire_faits
    from src.nlp_transformers import generer_tout
    from src.save_to_postgres import sauvegarder_tout
    from src.generate_pdf import generer_pdf

    # Lecture depuis PostgreSQL (pour API/PDF)
    from src.db import get_kpi_global, get_kpi_annual, get_anomalies
"""

__version__ = "4.0.0"
__author__ = "NLP Reporting Team"

# ── IMPORTANT : pas d'imports eagerly ici pour éviter les circular imports.
# ── Chaque module importe ce dont il a besoin directement.
# ── Voir l'usage dans le docstring ci-dessus.

__all__ = [
    "create_spark_session",
    "load_data",
    "compute_kpis",
    "collect_kpis",
    "extraire_faits",
    "NLTKProcessor",
    "generer_tout",
    "sauvegarder_tout",
    "generer_pdf",
]