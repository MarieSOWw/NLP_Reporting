"""
main.py - Point d'entree unique du pipeline NLP Reporting
============================================================

Role
----
Orchestre le pipeline complet en une seule commande. Coordonne les 7
etapes du flux end-to-end :
    1. Chargement et nettoyage des donnees (Spark)
    2. Calcul des 12 KPIs business (Spark)
    3. Extraction des faits via NLTK
    4. Generation du rapport NLP global (Mistral, 2 etapes)
    5. Sauvegarde PostgreSQL (KPIs + rapport global + ventes_detail)
    6. Generation des 11 rapports filtres (4 annees + 4 regions +
       3 categories), chacun avec ses faits recalcules depuis Postgres
    7. Export PDF du rapport global

Usage
-----
    python main.py                    pipeline complet francais (par defaut)
    python main.py --langue en        pipeline en anglais
    python main.py --skip-db          sans PostgreSQL (juste console + PDF)
    python main.py --skip-pdf         sans PDF (juste console + DB)
    python main.py --skip-db --skip-pdf   minimal (juste NLP)

Decisions de pipeline
---------------------
- Les rapports filtres recalculent leurs faits depuis ventes_detail
  pour etre coherents avec leur perimetre (pas une copie du global).
- L'analyse NLTK enrichie est persistee pour chacun des 12 rapports,
  effectuee sur le VRAI texte Mistral genere, pas sur un template.
- La croissance YoY = None quand pas d'annee comparable (au lieu
  d'inventer une variation intra-annee).
- ventes_detail est charge APRES la sauvegarde KPI globale et AVANT
  les rapports filtres (pour les agregations dynamiques).

Helpers
-------
- _charger_ventes_detail() : essaye plusieurs noms de fonctions selon
  la version de load_detail_table.py presente.
- _get_build_faits_from_pg() : importe la fonction d'assemblage des
  faits depuis api.py (source canonique) ou generate_pdf.py (fallback).
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if sys.platform == "win32":
    os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME", r"C:\hadoop")

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


from src.load_data import create_spark_session, load_data, afficher_apercu
from src.analytics import compute_kpis, afficher_kpis, collect_kpis
from src.nlp_nltk import extraire_faits, afficher_faits
from src.nlp_transformers import generer_tout, afficher_rapport_complet
from src.save_to_postgres import (
    sauvegarder_tout, get_connection, sauvegarder_rapport, creer_tables,
    sauvegarder_nltk_analysis,
)
from src.generate_pdf import generer_pdf


def banniere():
    print("\n" + "=" * 70)
    print("   NLP BUSINESS REPORTING - Pipeline Complet")
    print("   Spark -> NLTK -> Mistral API -> PostgreSQL -> PDF")
    print("=" * 70 + "\n")


def separator(step: int, total: int, title: str):
    print(f"\n{'-' * 70}")
    print(f"  [{step}/{total}] {title}")
    print(f"{'-' * 70}")


def _charger_ventes_detail():
    """
    Charge la table ventes_detail (granulaire) qui sert de source de
    verite pour tous les filtres dynamiques. Doit etre appelee AVANT
    les rapports filtres.

    Compatibilite : essaye plusieurs noms de fonction selon la version
    de load_detail_table.py.
    """
    try:
        from load_detail_table import load_csv, create_table
        conn = get_connection()
        try:
            create_table(conn)
            count = load_csv(conn)
            logger.info(f"[OK] Table ventes_detail chargee ({count} lignes)")
        finally:
            conn.close()
        return
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Chargement via load_csv echoue: {e}")

    try:
        from load_detail_table import charger_ventes_detail as _charger
        _charger()
        logger.info("[OK] Table ventes_detail chargee (fallback)")
        return
    except ImportError:
        pass

    logger.warning(
        "load_detail_table introuvable - les rapports filtres risquent d'etre incoherents"
    )


def run(langue: str = "fr", skip_db: bool = False, skip_pdf: bool = False) -> dict:
    """Execute le pipeline complet."""
    banniere()
    t_total = time.time()
    total_steps = 7

    # Etape 1 : chargement Spark
    separator(1, total_steps, "Chargement des donnees (Spark)")
    t = time.time()
    try:
        spark = create_spark_session()
        df = load_data(spark)
        afficher_apercu(df)
        logger.info(f"Duree: {time.time() - t:.1f}s")
    except Exception as e:
        logger.error(f"[KO] Erreur chargement: {e}")
        raise

    # Etape 2 : calcul des KPIs
    separator(2, total_steps, "Calcul des KPIs business (Spark)")
    t = time.time()
    try:
        kpis = compute_kpis(df)
        afficher_kpis(kpis, nb_rows=5)
        logger.info(f"Duree: {time.time() - t:.1f}s")
    except Exception as e:
        logger.error(f"[KO] Erreur KPIs: {e}")
        spark.stop()
        raise

    # Etape 3 : extraction NLTK
    separator(3, total_steps, "Extraction des faits (NLTK)")
    t = time.time()
    try:
        faits = extraire_faits(kpis)
        afficher_faits(faits)
        logger.info(f"Duree: {time.time() - t:.1f}s")
    except Exception as e:
        logger.error(f"[KO] Erreur NLTK: {e}")
        spark.stop()
        raise

    # Collecte des KPIs en memoire AVANT d'arreter Spark
    logger.info("Collecte des KPIs en memoire...")
    try:
        kpis_collected = collect_kpis(kpis)
        logger.info(f"[OK] {len(kpis_collected)} KPIs collectes")
    except Exception as e:
        logger.error(f"[KO] Erreur collecte: {e}")
        spark.stop()
        raise

    try:
        spark.stop()
        logger.info("Spark arrete")
    except Exception:
        pass

    # Etape 4 : generation NLP (Mistral)
    separator(4, total_steps, f"Generation du rapport NLP global ({langue.upper()})")
    t = time.time()
    try:
        resultats = generer_tout(faits, langue=langue)
        afficher_rapport_complet(
            resultats["rapport"],
            resultats["resume"],
            resultats["anomalies"],
            resultats["score"],
            langue=langue,
            structure=resultats.get("structure"),
            score_nlp=resultats.get("score_nlp"),
        )
        logger.info(f"Duree: {time.time() - t:.1f}s")
    except Exception as e:
        logger.error(f"[KO] Erreur NLP: {e}")
        if "resultats" not in locals():
            raise
        else:
            logger.warning("Affichage echoue mais resultats NLP disponibles, on continue.")

    # Etape 5 : sauvegarde PostgreSQL
    separator(5, total_steps, "Sauvegarde PostgreSQL (KPIs + rapport global)")
    nb_rapports = 0

    if not skip_db:
        t = time.time()
        try:
            sauvegarder_tout(kpis_collected, faits, resultats, report_type="global")
            nb_rapports += 1

            logger.info("Chargement de la table ventes_detail...")
            _charger_ventes_detail()

            logger.info(f"Duree etape 5: {time.time() - t:.1f}s")
        except Exception as e:
            logger.warning(f"PostgreSQL ignore: {e}")
            logger.info("Utiliser --skip-db pour desactiver cette etape")
    else:
        logger.info("PostgreSQL ignore (--skip-db)")

    # Etape 6 : rapports filtres
    separator(6, total_steps, "Rapports filtres (annee / region / categorie)")

    if not skip_db:
        t = time.time()
        build_faits_from_pg = _get_build_faits_from_pg()

        if build_faits_from_pg is None:
            logger.warning("_build_faits_from_pg indisponible - rapports filtres sautes")
        else:
            conn = get_connection()
            try:
                creer_tables(conn)

                # Rapports par annee
                annees_disponibles = sorted({a["annee"] for a in faits["annuel"]})
                for annee in annees_disponibles:
                    logger.info(f"Generation rapport annee {annee}...")
                    try:
                        faits_annee = build_faits_from_pg(year=annee)
                        resultats_annee = generer_tout(faits_annee, langue=langue)
                        sauvegarder_rapport(
                            conn, resultats_annee, faits_annee,
                            report_type="by_year", filter_year=annee,
                        )
                        sauvegarder_nltk_analysis(
                            conn,
                            rapport_texte=resultats_annee.get("rapport", ""),
                            langue=langue,
                            report_type="by_year",
                            filter_year=annee,
                        )
                        nb_rapports += 1
                        ca_annee = faits_annee["annuel"][0]["ca"] if faits_annee.get("annuel") else 0
                        logger.info(f"   [OK] Rapport {annee} sauvegarde (CA={ca_annee:,.0f}$)")
                    except Exception as e:
                        logger.warning(f"   Rapport {annee} echoue: {e}")
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                # Rapports par region
                for region in ["West", "East", "Central", "South"]:
                    logger.info(f"Generation rapport region {region}...")
                    try:
                        faits_region = build_faits_from_pg(region=region)
                        resultats_region = generer_tout(faits_region, langue=langue)
                        sauvegarder_rapport(
                            conn, resultats_region, faits_region,
                            report_type="by_region", filter_region=region,
                        )
                        sauvegarder_nltk_analysis(
                            conn,
                            rapport_texte=resultats_region.get("rapport", ""),
                            langue=langue,
                            report_type="by_region",
                            filter_region=region,
                        )
                        nb_rapports += 1
                        ca_region = sum(a["ca"] for a in faits_region.get("annuel", []))
                        logger.info(f"   [OK] Rapport {region} sauvegarde (CA region={ca_region:,.0f}$)")
                    except Exception as e:
                        logger.warning(f"   Rapport {region} echoue: {e}")
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                # Rapports par categorie
                for categorie in ["Furniture", "Office Supplies", "Technology"]:
                    logger.info(f"Generation rapport categorie {categorie}...")
                    try:
                        faits_cat = build_faits_from_pg(category=categorie)
                        resultats_cat = generer_tout(faits_cat, langue=langue)
                        sauvegarder_rapport(
                            conn, resultats_cat, faits_cat,
                            report_type="by_category", filter_category=categorie,
                        )
                        sauvegarder_nltk_analysis(
                            conn,
                            rapport_texte=resultats_cat.get("rapport", ""),
                            langue=langue,
                            report_type="by_category",
                            filter_category=categorie,
                        )
                        nb_rapports += 1
                        ca_cat = sum(a["ca"] for a in faits_cat.get("annuel", []))
                        logger.info(f"   [OK] Rapport {categorie} sauvegarde (CA cat={ca_cat:,.0f}$)")
                    except Exception as e:
                        logger.warning(f"   Rapport {categorie} echoue: {e}")
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                conn.commit()
                logger.info(f"Duree etape 6: {time.time() - t:.1f}s")
            except Exception as e:
                conn.rollback()
                logger.warning(f"Erreur globale rapports filtres: {e}")
            finally:
                conn.close()
    else:
        logger.info("Rapports filtres ignores (--skip-db)")

    # Etape 7 : export PDF du rapport global
    separator(7, total_steps, "Export PDF (rapport global)")
    chemin_pdf = None
    if not skip_pdf:
        t = time.time()
        try:
            chemin_pdf = generer_pdf(faits, resultats, langue=langue)
            logger.info(f"PDF: {chemin_pdf}")
            logger.info(f"Duree: {time.time() - t:.1f}s")
        except Exception as e:
            logger.warning(f"PDF ignore: {e}")
    else:
        logger.info("PDF ignore (--skip-pdf)")

    # Resume final
    duree_totale = time.time() - t_total

    print("\n" + "=" * 70)
    print("   PIPELINE TERMINE AVEC SUCCES")
    print("=" * 70)
    print("\n  Resultats:")
    print(f"     Langue        : {langue.upper()}")
    print(f"     Score perf    : {resultats['score']['score']}/100 - {resultats['score']['mention']}")
    if resultats.get("score_nlp"):
        sn = resultats["score_nlp"]
        print(f"     Score NLP     : {sn.get('score_nlp', 0)}/100 - {sn.get('mention', '')}")
    print(f"     Anomalies     : {len(resultats['anomalies'])}")

    cg = faits.get("croissance_globale")
    if cg is not None:
        try:
            print(f"     Croissance    : {float(cg):+.1f}%")
        except Exception:
            print("     Croissance    : N/A")
    else:
        print("     Croissance    : N/A (premiere annee)")
    print(f"     Periode       : {faits['periode']}")

    if chemin_pdf:
        print("\n  Fichiers generes:")
        print(f"     PDF: {chemin_pdf}")

    if not skip_db:
        print("\n  PostgreSQL:")
        print("     kpi_global + 7 tables KPI + ventes_detail")
        print(
            f"     {nb_rapports} rapports NLP sauvegardes "
            f"(1 global + {nb_rapports - 1} filtres)"
        )
        print(f"     {nb_rapports} analyses NLTK sauvegardees (chacune sur le vrai rapport Mistral)")

    print(f"\n  Duree totale: {duree_totale:.1f}s")
    print("=" * 70 + "\n")

    return resultats


def _get_build_faits_from_pg():
    """
    Renvoie la fonction _build_faits_from_pg.
    Priorite : api.py (source canonique) > generate_pdf.py (fallback).
    """
    try:
        from api import _build_faits_from_pg
        return _build_faits_from_pg
    except Exception:
        pass
    try:
        from src.generate_pdf import _build_faits_from_pg
        return _build_faits_from_pg
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="NLP Reporting - Pipeline complet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py                    # Pipeline complet en francais
  python main.py --langue en        # Pipeline en anglais
  python main.py --skip-db          # Sans PostgreSQL
  python main.py --skip-pdf         # Sans PDF
  python main.py --skip-db --skip-pdf  # Minimal (juste NLP)
        """,
    )
    parser.add_argument("--langue", "-l", choices=["fr", "en"], default="fr",
                        help="Langue du rapport genere (defaut: fr)")
    parser.add_argument("--skip-db", action="store_true",
                        help="Ne pas sauvegarder dans PostgreSQL")
    parser.add_argument("--skip-pdf", action="store_true",
                        help="Ne pas generer le PDF")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mode verbose (plus de details)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        run(langue=args.langue, skip_db=args.skip_db, skip_pdf=args.skip_pdf)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[KO] Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
