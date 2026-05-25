"""
analytics.py - Calcul des KPIs business avec Spark
====================================================

Role
----
Deuxieme maillon du pipeline. Prend le DataFrame nettoye produit par
load_data.py et calcule les 12 indicateurs business attendus. Chaque
KPI est un DataFrame Spark separe, stocke dans le dict `kpis`.

Pourquoi ces 12 KPIs
--------------------
Ils couvrent les axes d'analyse standard d'un livrable executif :
    1. region_trim       : ventes par region et trimestre
    2. categorie         : ventes par categorie et region
    3. top_produits      : top 10 sous-categories
    4. variation         : variations trimestrielles QoQ + YoY
    5. annuel            : performance annuelle avec YoY
    6. meilleure_region  : meilleure region par annee
    7. mensuel           : ventes mensuelles + moyenne mobile 3M
    8. segment           : ventes par segment client
    9. ship_mode         : performance par mode de livraison
   10. top_clients       : top 10 clients (RFM simplifie)
   11. etats             : analyse par etat americain
   12. saisonnalite      : par jour de la semaine

Definitions metier (canoniques, utilisees aussi en BDD et en API)
-----------------------------------------------------------------
- Nb_Commandes     : countDistinct(Order_ID)  - vraies commandes
- Nb_Articles      : count(*)                  - lignes du CSV
- Panier_Moyen     : SUM(Sales) / Nb_Commandes - panier par commande
- Croissance YoY   : variation par rapport a l'annee precedente
- Variation QoQ    : variation par rapport au trimestre precedent
- Variation YoY    : variation meme trimestre annee precedente
                     (utilisee pour la detection d'anomalies, evite les
                     faux positifs saisonniers)

Architecture Spark
------------------
- Toutes les agregations utilisent groupBy + agg.
- Les variations utilisent des Window functions avec lag().
- La moyenne mobile 3M utilise une Window de 3 lignes glissantes.
- Toutes les agregations financieres sont arrondies a 2 decimales.

Bonus : sauvegarde Parquet partitionnee (save_kpis_to_parquet) pour
demontrer la maitrise du stockage Spark colonnaire.

Sorties
-------
- compute_kpis(df)              : dict de DataFrames Spark
- afficher_kpis(kpis)           : impression console
- collect_kpis(kpis)            : conversion en listes Python (a appeler AVANT spark.stop)
- save_kpis_to_parquet(kpis)    : ecriture Parquet partitionnee (optionnel)
"""

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if sys.platform == "win32":
    os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME", r"C:\hadoop")

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.load_data import create_spark_session, load_data

try:
    from config import BASE_DIR
    PARQUET_DIR = BASE_DIR / "data" / "parquet"
except ImportError:
    PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_kpis(df) -> dict:
    """Calcule les 12 KPIs business. Renvoie un dict de DataFrames Spark."""
    logger.info("Calcul des KPIs business...")
    df = df.filter(F.col("Annee").isNotNull())
    kpis = {}

    kpis["region_trim"] = (
        df.groupBy("Region", "Annee", "Trimestre")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Totales"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
            F.count("*").alias("Nb_Articles"),
            F.round(F.sum("Sales") / F.countDistinct("Order_ID"), 2).alias("Panier_Moyen"),
            F.countDistinct("Customer_ID").alias("Clients_Uniques"),
        )
        .orderBy("Region", "Annee", "Trimestre")
    )

    kpis["categorie"] = (
        df.groupBy("Region", "Category", "Annee")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Categorie"),
            F.count("*").alias("Nb_Articles"),
            F.round(F.avg("Sales"), 2).alias("Prix_Moyen_Article"),
        )
        .orderBy("Annee", "Region", F.desc("Ventes_Categorie"))
    )

    kpis["top_produits"] = (
        df.groupBy("Sub_Category")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Totales"),
            F.count("*").alias("Quantite_Vendue"),
            F.round(F.avg("Sales"), 2).alias("Prix_Moyen_Article"),
            F.countDistinct("Customer_ID").alias("Nb_Clients"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
        )
        .orderBy(F.desc("Ventes_Totales"))
        .limit(10)
    )

    ventes_trim = (
        df.groupBy("Annee", "Trimestre")
        .agg(F.round(F.sum("Sales"), 2).alias("Ventes_Totales"))
        .withColumn("Periode", F.col("Annee") * 10 + F.col("Trimestre"))
        .orderBy("Annee", "Trimestre")
    )
    window_qoq = Window.orderBy("Periode")
    kpis["variation"] = (
        ventes_trim
        .withColumn("Ventes_Precedentes_QoQ", F.lag("Ventes_Totales", 1).over(window_qoq))
        .withColumn(
            "Variation_QoQ_Pct",
            F.round(
                (F.col("Ventes_Totales") - F.col("Ventes_Precedentes_QoQ"))
                / F.col("Ventes_Precedentes_QoQ") * 100,
                2,
            ),
        )
    )
    window_yoy = Window.partitionBy("Trimestre").orderBy("Annee")
    kpis["variation"] = (
        kpis["variation"]
        .withColumn(
            "Ventes_Meme_Trim_An_Precedent",
            F.lag("Ventes_Totales", 1).over(window_yoy),
        )
        .withColumn(
            "Variation_YoY_Pct",
            F.round(
                (F.col("Ventes_Totales") - F.col("Ventes_Meme_Trim_An_Precedent"))
                / F.col("Ventes_Meme_Trim_An_Precedent") * 100,
                2,
            ),
        )
    )
    kpis["variation"] = (
        kpis["variation"]
        .withColumn(
            "Croissance_Cumul",
            F.round(
                (F.col("Ventes_Totales") - F.first("Ventes_Totales").over(Window.orderBy("Periode")))
                / F.first("Ventes_Totales").over(Window.orderBy("Periode")) * 100,
                2,
            ),
        )
        .drop("Periode")
    )
    kpis["variation"] = kpis["variation"].withColumn("Variation_Pct", F.col("Variation_QoQ_Pct"))

    kpis["annuel"] = (
        df.groupBy("Annee")
        .agg(
            F.round(F.sum("Sales"), 2).alias("CA_Annuel"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
            F.count("*").alias("Nb_Articles"),
            F.round(F.sum("Sales") / F.countDistinct("Order_ID"), 2).alias("Panier_Moyen"),
            F.countDistinct("Customer_ID").alias("Clients_Uniques"),
            F.countDistinct("Product_ID").alias("Produits_Vendus"),
        )
        .orderBy("Annee")
    )
    window_annuel = Window.orderBy("Annee")
    kpis["annuel"] = (
        kpis["annuel"]
        .withColumn("CA_Precedent", F.lag("CA_Annuel", 1).over(window_annuel))
        .withColumn(
            "Croissance_YoY",
            F.round(
                (F.col("CA_Annuel") - F.col("CA_Precedent"))
                / F.col("CA_Precedent") * 100,
                2,
            ),
        )
    )

    kpis["meilleure_region"] = (
        df.groupBy("Annee", "Region")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Totales"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
        )
        .orderBy("Annee", F.desc("Ventes_Totales"))
    )

    kpis["mensuel"] = (
        df.groupBy("Annee", "Mois")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Mensuelles"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
            F.round(F.sum("Sales") / F.countDistinct("Order_ID"), 2).alias("Panier_Moyen"),
        )
        .orderBy("Annee", "Mois")
    )
    window_mobile = Window.orderBy("Annee", "Mois").rowsBetween(-2, 0)
    kpis["mensuel"] = kpis["mensuel"].withColumn(
        "Moyenne_Mobile_3M",
        F.round(F.avg("Ventes_Mensuelles").over(window_mobile), 2),
    )

    kpis["segment"] = (
        df.groupBy("Segment", "Annee")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Segment"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
            F.round(F.sum("Sales") / F.countDistinct("Order_ID"), 2).alias("Panier_Moyen"),
            F.countDistinct("Customer_ID").alias("Nb_Clients"),
        )
        .orderBy("Annee", F.desc("Ventes_Segment"))
    )

    kpis["ship_mode"] = (
        df.groupBy("Ship_Mode", "Annee")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Totales"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
            F.round(F.avg("Delai_Livraison"), 1).alias("Delai_Moyen_Jours"),
        )
        .orderBy("Annee", F.desc("Ventes_Totales"))
    )

    kpis["top_clients"] = (
        df.groupBy("Customer_ID", "Customer_Name")
        .agg(
            F.round(F.sum("Sales"), 2).alias("CA_Total"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
            F.round(F.sum("Sales") / F.countDistinct("Order_ID"), 2).alias("Panier_Moyen"),
            F.max("Order_Date").alias("Derniere_Commande"),
        )
        .orderBy(F.desc("CA_Total"))
        .limit(10)
    )

    kpis["etats"] = (
        df.groupBy("State", "Region")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Totales"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
            F.countDistinct("Customer_ID").alias("Nb_Clients"),
        )
        .orderBy(F.desc("Ventes_Totales"))
        .limit(15)
    )

    kpis["saisonnalite"] = (
        df.groupBy("Jour_Semaine")
        .agg(
            F.round(F.sum("Sales"), 2).alias("Ventes_Totales"),
            F.countDistinct("Order_ID").alias("Nb_Commandes"),
            F.round(F.sum("Sales") / F.countDistinct("Order_ID"), 2).alias("Panier_Moyen"),
        )
        .orderBy("Jour_Semaine")
    )

    logger.info(f"[OK] {len(kpis)} KPIs calcules")
    return kpis


def save_kpis_to_parquet(kpis: dict, output_dir: Path = None):
    """Sauvegarde les KPIs en Parquet partitionne (Spark colonnaire)."""
    output_dir = Path(output_dir or PARQUET_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ecriture Parquet partitionne dans {output_dir}...")

    partition_map = {
        "annuel":           ["Annee"],
        "region_trim":      ["Annee", "Region"],
        "categorie":        ["Annee", "Region"],
        "variation":        ["Annee"],
        "meilleure_region": ["Annee"],
        "mensuel":          ["Annee"],
        "segment":          ["Annee"],
        "ship_mode":        ["Annee"],
        "etats":            ["Region"],
        "top_produits":     [],
        "top_clients":      [],
        "saisonnalite":     [],
    }

    for key, df in kpis.items():
        target = output_dir / key
        partitions = partition_map.get(key, [])
        try:
            writer = df.write.mode("overwrite").option("compression", "snappy")
            if partitions:
                writer = writer.partitionBy(*partitions)
            writer.parquet(str(target))
            logger.info(f"   {key:20s} -> {target.name}/ (partitions: {partitions or 'none'})")
        except Exception as e:
            logger.warning(f"   {key}: {e}")

    logger.info("[OK] Parquet ecrit")
    return output_dir


def afficher_kpis(kpis: dict, nb_rows: int = 8):
    """Imprime tous les KPIs dans la console."""
    print("\n" + "=" * 65)
    print("   KPIs BUSINESS CALCULES PAR SPARK")
    print("=" * 65)
    kpi_labels = {
        "region_trim": "[KPI 1] Ventes par region et trimestre",
        "categorie": "[KPI 2] Ventes par categorie et region",
        "top_produits": "[KPI 3] Top 10 sous-categories",
        "variation": "[KPI 4] Variations trimestrielles (QoQ)",
        "annuel": "[KPI 5] Performance annuelle avec YoY",
        "meilleure_region": "[KPI 6] Meilleure region par annee",
        "mensuel": "[KPI 7] Ventes mensuelles + moyenne mobile",
        "segment": "[KPI 8] Ventes par segment client",
        "ship_mode": "[KPI 9] Performance par mode de livraison",
        "top_clients": "[KPI 10] Top 10 clients",
        "etats": "[KPI 11] Top 15 Etats",
        "saisonnalite": "[KPI 12] Saisonnalite (jour de semaine)",
    }
    for key, label in kpi_labels.items():
        if key in kpis:
            print(f"\n{label}:")
            rows = 10 if key in ["top_produits", "top_clients"] else nb_rows
            kpis[key].show(rows, truncate=False)
    print("=" * 65)


def collect_kpis(kpis: dict) -> dict:
    """Convertit les DataFrames Spark en listes Python. A appeler AVANT spark.stop()."""
    logger.info("Collecte des KPIs en memoire...")
    kpis_collected = {}
    for key, df in kpis.items():
        kpis_collected[key] = df.collect()
        logger.info(f"   {key}: {len(kpis_collected[key])} lignes")
    return kpis_collected


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-parquet", action="store_true")
    args = parser.parse_args()

    try:
        spark = create_spark_session()
        df = load_data(spark)
        kpis = compute_kpis(df)
        afficher_kpis(kpis)
        if args.save_parquet:
            save_kpis_to_parquet(kpis)
        spark.stop()
        print("\n[OK] Etape 2 - analytics.py terminee")
    except Exception as e:
        logger.error(f"[KO] Erreur: {e}")
        raise
