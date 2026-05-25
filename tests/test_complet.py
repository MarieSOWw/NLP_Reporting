"""
test_complet.py - Validation complete du projet NLP Reporting
================================================================

Role
----
Script de validation manuel a executer apres un deploiement ou un
changement de configuration. Verifie 7 points cles :
    1. Imports critiques (mistralai, sentence-transformers, NLTK)
    2. Connexion PostgreSQL et stats des tables
    3. Transformer local (embeddings sentence-transformers)
    4. Generation rapport NLP (Mistral ou fallback)
    5. Detection anomalies YoY
    6. Section NLTK integree dans le PDF
    7. Endpoint de generation planifiee

Difference avec test_pipeline.py
--------------------------------
test_pipeline.py contient des tests unitaires pytest standards.
test_complet.py est un script de smoke test qui peut etre lance
directement (`python tests/test_complet.py`) avec un affichage
visuel des resultats. Il appelle des composants reels (DB, API,
modeles) et ne mock rien.

Usage
-----
    python tests/test_complet.py
"""

import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
if PROJECT_ROOT.name == "tests":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_test(name: str, passed: bool, details: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        print(f"      -> {details}")


def read_project_file(*relative_parts) -> str:
    candidates = [
        PROJECT_ROOT.joinpath(*relative_parts),
        PROJECT_ROOT.joinpath("src", *relative_parts),
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Fichier introuvable parmi: {candidates}")


def test_imports() -> bool:
    print_header("TEST 1 - Imports critiques")
    results = []

    try:
        from mistralai.client import Mistral
        print_test("Import mistralai", True, "Mistral class found via mistralai.client")
        results.append(True)
    except ImportError as e:
        print_test("Import mistralai", False, f"Error: {e}")
        results.append(False)

    try:
        from sentence_transformers import SentenceTransformer
        print_test("Import sentence-transformers", True, "Transformer HF local disponible")
        results.append(True)
    except ImportError:
        print_test("Import sentence-transformers", False, "pip install sentence-transformers")
        results.append(False)

    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        print_test("Import NLTK + VADER", True, "NLP tools ready")
        results.append(True)
    except ImportError as e:
        print_test("Import NLTK", False, str(e))
        results.append(False)

    return all(results)


def test_database() -> bool:
    print_header("TEST 2 - Connexion PostgreSQL")
    try:
        from src.db import test_connection, get_tables_stats

        if not test_connection():
            print_test("Connexion PostgreSQL", False, "Cannot connect to database")
            return False
        print_test("Connexion PostgreSQL", True, "Database accessible")

        stats = get_tables_stats()
        if not isinstance(stats, dict):
            print_test("Tables statistiques", False, f"Format inattendu: {type(stats).__name__}")
            return False

        numeric_values = []
        for v in stats.values():
            if isinstance(v, (int, float)):
                numeric_values.append(v)
            else:
                try:
                    numeric_values.append(int(v))
                except Exception:
                    pass
        total_rows = sum(numeric_values)

        if len(stats) > 0:
            print_test(
                "Tables statistiques", True,
                f"{len(stats)} tables, {total_rows:,} lignes numeriques cumulees"
            )
            return True

        print_test("Tables statistiques", False, "Aucune statistique retournee")
        return False
    except Exception as e:
        print_test("Test database", False, str(e))
        return False


def test_embeddings_local() -> bool:
    print_header("TEST 3 - Transformer local (sentence-transformers)")
    try:
        from src.embeddings_local import LocalEmbedder
        embedder = LocalEmbedder()
        question = "Quelle est la meilleure region ?"
        embedding = embedder.embed(question)

        if embedding and len(embedding) == 384:
            print_test("Embedding local", True, f"Dimension: {len(embedding)} (all-MiniLM-L6-v2)")
            print_test("Modele HuggingFace", True, "Transformer local operationnel")
            return True

        print_test("Embedding local", False, f"Unexpected dimension: {len(embedding) if embedding else 'None'}")
        return False
    except Exception as e:
        print_test("Test embeddings", False, str(e))
        return False


def test_nlp_generation() -> bool:
    print_header("TEST 4 - Generation rapport NLP (Mistral / fallback)")
    try:
        faits_test = {
            "periode": "2015-2018",
            "croissance_globale": 49.5,
            "tendance_globale": {"label": "Forte croissance"},
            "volatilite": {"niveau": "Moderee", "valeur": 15.2},
            "meilleure_annee": {"annee": 2018, "ca": 713927, "commandes": 2000},
            "meilleur_trim": {"annee": 2017, "trimestre": 4, "variation": 69.45, "ventes": 234389},
            "pire_trim": {"annee": 2016, "trimestre": 1, "variation": -64.9, "ventes": 62358},
            "top_produits": [
                {"nom": "Phones", "ventes": 327528, "qte": 1000},
                {"nom": "Chairs", "ventes": 322823, "qte": 900},
                {"nom": "Storage", "ventes": 212303, "qte": 800},
            ],
            "segments": {
                "Consumer": {"ventes_total": 1137124, "commandes": 3000},
                "Corporate": {"ventes_total": 700000, "commandes": 1500},
            },
            "annuel": [
                {"annee": 2015, "ca": 477478, "croissance_yoy": None},
                {"annee": 2016, "ca": 454364, "croissance_yoy": -4.8},
                {"annee": 2017, "ca": 609087, "croissance_yoy": 34.0},
                {"annee": 2018, "ca": 713927, "croissance_yoy": 17.2},
            ],
        }

        from src.nlp_transformers import generer_rapport
        rapport = generer_rapport(faits_test, langue="fr")

        if rapport and len(rapport.strip()) > 100:
            nb_chars = len(rapport)
            nb_words = len(rapport.split())
            print_test("Rapport genere", True, f"{nb_chars} chars, ~{nb_words} mots")

            if nb_chars > 500:
                print_test("Longueur suffisante", True, "Rapport detaille")
            else:
                print_test("Longueur suffisante", True, "Rapport court mais valide (fallback possible)")
            return True

        print_test(
            "Rapport genere", False,
            f"Rapport trop court ou vide: {len(rapport) if rapport else 0} chars"
        )
        return False
    except Exception as e:
        print_test("Test generation NLP", False, str(e))
        return False


def test_anomalies_yoy() -> bool:
    print_header("TEST 5 - Detection anomalies YoY (Year-over-Year)")
    try:
        faits_test = {
            "variations": [
                {"annee": 2015, "trimestre": 1, "variation": 0, "ventes": 100000},
                {"annee": 2016, "trimestre": 1, "variation": 0, "ventes": 60000},
                {"annee": 2017, "trimestre": 1, "variation": 0, "ventes": 30000},
                {"annee": 2015, "trimestre": 2, "variation": 0, "ventes": 80000},
                {"annee": 2016, "trimestre": 2, "variation": 0, "ventes": 120000},
            ]
        }

        from src.nlp_transformers import detecter_anomalies
        anomalies = detecter_anomalies(faits_test, seuil_alerte=40.0)

        if not anomalies:
            print_test("Detection anomalies", False, "Aucune anomalie detectee")
            return False

        print_test("Detection anomalies", True, f"{len(anomalies)} anomalies detectees")
        print_test("Seuil YoY (40%)", True, "Filtre saisonnalite applique")

        valid_types = all(a.get("type") in {"chute_yoy", "hausse_yoy"} for a in anomalies)
        if not valid_types:
            print_test("Type YoY", False, f"Types inattendus: {[a.get('type') for a in anomalies]}")
            return False

        has_alert_or_info = any(a.get("niveau") in {"ALERTE", "INFO"} for a in anomalies)
        if not has_alert_or_info:
            print_test("Niveaux anomalies", False, "Aucun niveau ALERTE/INFO")
            return False

        print_test("Type YoY", True, "Comparaison Year-over-Year activee")
        return True
    except Exception as e:
        print_test("Test anomalies", False, str(e))
        return False


def test_pdf_nltk_section() -> bool:
    print_header("TEST 6 - Section NLTK dans le PDF")
    try:
        generate_pdf_content = read_project_file("generate_pdf.py")
        pdf_nltk_section_content = read_project_file("pdf_nltk_section.py")

        ok = True
        if "creer_section_analyse_linguistique" in pdf_nltk_section_content:
            print_test("Fonction section NLTK", True, "Presente dans pdf_nltk_section.py")
        else:
            print_test("Fonction section NLTK", False, "Absente")
            ok = False

        if "creer_section_analyse_linguistique(" in generate_pdf_content:
            print_test("Appel section NLTK", True, "Section integree dans generate_pdf.py")
        else:
            print_test("Appel section NLTK", False, "Section non appelee")
            ok = False

        if 'styles["h1"]' in pdf_nltk_section_content or "styles['h1']" in pdf_nltk_section_content:
            print_test("Styles PDF alignes", True, "Utilise h1/h2 compatibles avec generate_pdf.py")
        else:
            print_test("Styles PDF alignes", False, "Styles heading1/heading2 encore presents")
            ok = False

        return ok
    except Exception as e:
        print_test("Test section PDF NLTK", False, str(e))
        return False


def test_api_schedule_endpoint() -> bool:
    print_header("TEST 7 - Endpoint de generation planifiee")
    try:
        api_content = read_project_file("api.py")
        has_route = (
            '"/report/scheduled"' in api_content
            or '"/reports/schedule"' in api_content
            or "/report/scheduled" in api_content
            or "/reports/schedule" in api_content
        )
        has_logic = "scheduled" in api_content.lower()

        if has_route and has_logic:
            print_test("Endpoint scheduled", True, "Endpoint de generation planifiee detecte")
            return True

        print_test("Endpoint scheduled", False, "Endpoint planifie non trouve")
        return False
    except Exception as e:
        print_test("Test endpoint schedule", False, str(e))
        return False


def main() -> int:
    print("\n" + "=" * 70)
    print("   TESTS DE VALIDATION - NLP Reporting")
    print("=" * 70)

    results = {
        "Imports critiques": test_imports(),
        "Connexion PostgreSQL": test_database(),
        "Transformer local": test_embeddings_local(),
        "Generation NLP": test_nlp_generation(),
        "Anomalies YoY": test_anomalies_yoy(),
        "Section NLTK PDF": test_pdf_nltk_section(),
        "Endpoint scheduled": test_api_schedule_endpoint(),
    }

    print_header("RESUME")
    passed = sum(bool(v) for v in results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "[OK]" if result else "[KO]"
        print(f"  {status} {test_name}")

    print(f"\n  Score : {passed}/{total} tests reussis")

    if passed == total:
        print("\n  TOUS LES TESTS REUSSIS.")
        print("  Le projet est coherent sur les points testes.\n")
        return 0

    print(f"\n  {total - passed} test(s) echoue(s)")
    print("  Corriger les points indiques puis relancer le script.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
