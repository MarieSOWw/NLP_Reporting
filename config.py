"""
config.py - Configuration centralisee du projet NLP Reporting
==============================================================

Role
----
Point unique de configuration du projet. Toutes les valeurs sensibles
(connexion PostgreSQL, cle Mistral) sont lues via des variables
d'environnement charges depuis le fichier `.env` situe a la racine.
Tous les modules du projet importent depuis ici plutot que de lire
os.getenv directement, afin d'avoir une source unique de verite.

Ce fichier expose :
    - Chemins portables (BASE_DIR, DATA_DIR, OUTPUT_DIR, LOG_DIR, CACHE_DIR)
    - Parametres Spark (master, log level, hadoop home Windows)
    - Connexion PostgreSQL (host, port, dbname, user, password, URL SQLAlchemy)
    - Parametres FastAPI (host, port, CORS, prefix)
    - Cle Mistral et nom du modele (defaut : mistral-small-latest)
    - Liste des packages NLTK necessaires
    - Configuration PDF (chemins, auteur, titre)
    - Seuils analytiques (ANOMALY_THRESHOLD a 20%, top produits a 10)
    - Ponderation du score de performance business (SCORE_WEIGHTS)
    - Types de rapports supportes (global, by_year, by_region, by_category)
    - Langues supportees (fr, en)

Principes
---------
- Aucune valeur sensible n'est commitee : MISTRAL_API_KEY defaut a chaine vide.
- ANOMALY_THRESHOLD = 20 est la valeur canonique du projet, utilisee aussi
  bien cote PostgreSQL (db.get_anomalies) que cote Python (detecter_anomalies).
- API_PREFIX est utilise dans toutes les routes via f-strings, ce qui permet
  de le changer ici sans toucher au code des endpoints.

Usage
-----
    from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    python config.py        # imprime la config et valide les chemins
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Chemins portables (Windows, Linux, Mac)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "train.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


# Spark

SPARK_APP_NAME = "NLP_Business_Reporting"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_LOG_LEVEL = os.getenv("SPARK_LOG_LEVEL", "ERROR")
HADOOP_HOME = os.getenv("HADOOP_HOME", r"C:\hadoop")


# PostgreSQL

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "nlp_reporting")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin123")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# FastAPI

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"
API_CORS_ORIGINS = os.getenv("API_CORS_ORIGINS", "*").split(",")
API_PREFIX = "/api"


# Mistral (la cle DOIT venir du .env, pas de defaut hardcode)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


# NLTK

NLTK_DATA_DIR = CACHE_DIR / "nltk_data"
NLTK_PACKAGES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
    "vader_lexicon",
]


# PDF

PDF_OUTPUT = OUTPUT_DIR / "rapport_business.pdf"
PDF_AUTHOR = "NLP Reporting Pipeline"
PDF_TITLE = "Rapport Business Automatique - Superstore"


# Seuils analytiques (canoniques pour tout le projet)
# ANOMALY_THRESHOLD = 20% applique aux variations YoY dans :
#   - src/nlp_transformers.detecter_anomalies (Python)
#   - src/db.get_anomalies (PostgreSQL)
#   - dashboard Grafana

ANOMALY_THRESHOLD = 20.0
TOP_PRODUCTS_COUNT = 10

SCORE_WEIGHTS = {
    "croissance": 30,
    "regularite": 25,
    "meilleur_trim": 25,
    "diversite": 20,
}


# Types de rapports et langues supportes

REPORT_TYPES = ["global", "by_year", "by_region", "by_category"]
SUPPORTED_LANGUAGES = ["fr", "en"]


# Logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOG_DIR / "nlp_reporting.log"


def validate_config():
    """Verifie que les chemins critiques sont accessibles."""
    errors = []
    if not DATA_PATH.exists():
        errors.append(f"Fichier de donnees introuvable: {DATA_PATH}")
    for dir_path, dir_name in [(OUTPUT_DIR, "outputs"), (LOG_DIR, "logs")]:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Impossible de creer le repertoire {dir_name}: {e}")
    return errors


def print_config():
    """Affiche la configuration courante (debug)."""
    print("\n" + "=" * 60)
    print("   CONFIGURATION NLP REPORTING")
    print("=" * 60)
    print(f"  BASE_DIR       : {BASE_DIR}")
    print(f"  DATA_PATH      : {DATA_PATH}")
    print(f"  OUTPUT_DIR     : {OUTPUT_DIR}")
    print(f"  DB_HOST        : {DB_HOST}:{DB_PORT}")
    print(f"  DB_NAME        : {DB_NAME}")
    print(f"  MISTRAL_MODEL  : {MISTRAL_MODEL}")
    print(f"  MISTRAL_API_KEY: {'***' + MISTRAL_API_KEY[-4:] if MISTRAL_API_KEY else '(non configuree)'}")
    print(f"  SPARK_MASTER   : {SPARK_MASTER}")
    print(f"  API            : {API_HOST}:{API_PORT}")
    print(f"  API_PREFIX     : {API_PREFIX}")
    print(f"  CORS           : {API_CORS_ORIGINS}")
    print(f"  ANOMALY_THRES  : +/- {ANOMALY_THRESHOLD}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_config()
    errors = validate_config()
    if errors:
        print("[KO] Erreurs de configuration:")
        for err in errors:
            print(f"   - {err}")
    else:
        print("[OK] Configuration valide.")
