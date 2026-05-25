"""
load_data.py - Chargement et nettoyage des donnees avec Spark
==============================================================

Role
----
Premier maillon du pipeline. Cree une session Spark configuree pour
l'analyse locale, charge le CSV Superstore Sales, nettoie les donnees
(dates, sales invalides) et enrichit le DataFrame de colonnes
temporelles derivees (Annee, Trimestre, Mois, Semaine, Jour_Semaine,
Delai_Livraison).

Pourquoi Spark
--------------
Le dataset fait environ 9 800 lignes - pandas suffirait. Mais le
projet est ecrit pour passer a l'echelle sans reecriture : si demain
on passe a 9 millions de lignes, rien ne change dans le code metier
en aval. Spark est aussi utilise dans analytics.py pour les Window
functions (lag, rank, moyenne mobile).

Schema des donnees
------------------
SALES_SCHEMA decrit les 18 colonnes du CSV. Les types sont declares
explicitement pour plus de robustesse (au lieu de laisser Spark
inferer, ce qui peut diverger entre executions).

Configuration Windows
---------------------
HADOOP_HOME est forcee sur C:\\hadoop (winutils.exe) pour eviter les
erreurs Spark sous Windows. Le PATH inclut HADOOP_HOME\\bin.

Sorties
-------
- create_spark_session()  : SparkSession configuree
- load_data(spark)        : DataFrame nettoye avec colonnes temporelles
- get_dataset_stats(df)   : dict de statistiques descriptives
- afficher_apercu(df)     : impression console des stats

Cleaning applique
-----------------
- Filtrage des lignes avec Sales nulle ou <= 0
- Filtrage des lignes avec Order_Date invalide (format dd/MM/yyyy)
- Renommage des colonnes (espaces -> underscores)
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
    hadoop_home = os.getenv("HADOOP_HOME", r"C:\hadoop")
    os.environ["HADOOP_HOME"] = hadoop_home
    winutils_path = os.path.join(hadoop_home, "bin")
    if winutils_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = winutils_path + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType,
)

try:
    from config import DATA_PATH, SPARK_APP_NAME, SPARK_MASTER, SPARK_LOG_LEVEL
except ImportError:
    DATA_PATH = Path(__file__).parent / "data" / "train.csv"
    SPARK_APP_NAME = "NLP_Business_Reporting"
    SPARK_MASTER = "local[*]"
    SPARK_LOG_LEVEL = "ERROR"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SALES_SCHEMA = StructType([
    StructField("Row_ID", IntegerType(), True),
    StructField("Order_ID", StringType(), True),
    StructField("Order_Date", StringType(), True),
    StructField("Ship_Date", StringType(), True),
    StructField("Ship_Mode", StringType(), True),
    StructField("Customer_ID", StringType(), True),
    StructField("Customer_Name", StringType(), True),
    StructField("Segment", StringType(), True),
    StructField("Country", StringType(), True),
    StructField("City", StringType(), True),
    StructField("State", StringType(), True),
    StructField("Postal_Code", FloatType(), True),
    StructField("Region", StringType(), True),
    StructField("Product_ID", StringType(), True),
    StructField("Category", StringType(), True),
    StructField("Sub_Category", StringType(), True),
    StructField("Product_Name", StringType(), True),
    StructField("Sales", FloatType(), True),
])


def create_spark_session(app_name: str = None) -> SparkSession:
    """Cree une SparkSession optimisee pour l'analyse locale."""
    app_name = app_name or SPARK_APP_NAME
    logger.info(f"Initialisation de Spark Session: {app_name}")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(SPARK_MASTER)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
    )

    java_options = " ".join([
        "--add-opens=java.base/javax.security.auth=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
    ])
    builder = (
        builder
        .config("spark.driver.extraJavaOptions", java_options)
        .config("spark.executor.extraJavaOptions", java_options)
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(SPARK_LOG_LEVEL)
    logger.info("[OK] Spark Session creee")
    return spark


def load_data(spark: SparkSession, file_path: str = None):
    """
    Charge et nettoie le dataset Superstore Sales.

    Raises
    ------
    FileNotFoundError : fichier introuvable
    ValueError        : DataFrame vide apres nettoyage
    """
    file_path = str(file_path or DATA_PATH)
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Fichier de donnees introuvable: {file_path}")

    logger.info(f"Chargement des donnees depuis: {file_path}")
    df = spark.read.csv(file_path, header=True, inferSchema=True, encoding="utf-8")
    nb_initial = df.count()
    logger.info(f"   {nb_initial:,} lignes chargees")

    df = df.toDF(*[c.strip().replace(" ", "_").replace("-", "_") for c in df.columns])

    df = df.withColumn("Order_Date", F.to_date(F.col("Order_Date"), "dd/MM/yyyy"))
    df = df.withColumn("Ship_Date", F.to_date(F.col("Ship_Date"), "dd/MM/yyyy"))

    df_invalid_sales = df.filter(F.col("Sales").isNull() | (F.col("Sales") <= 0))
    nb_invalid_sales = df_invalid_sales.count()
    if nb_invalid_sales > 0:
        logger.warning(f"   {nb_invalid_sales} ligne(s) avec Sales invalide ignoree(s)")
        df = df.filter(F.col("Sales").isNotNull() & (F.col("Sales") > 0))

    df_invalid_dates = df.filter(F.col("Order_Date").isNull())
    nb_invalid_dates = df_invalid_dates.count()
    if nb_invalid_dates > 0:
        logger.warning(f"   {nb_invalid_dates} ligne(s) avec Order_Date invalide ignoree(s)")
        df = df.filter(F.col("Order_Date").isNotNull())

    df = df.withColumn("Annee", F.year("Order_Date"))
    df = df.withColumn("Trimestre", F.quarter("Order_Date"))
    df = df.withColumn("Mois", F.month("Order_Date"))
    df = df.withColumn("Semaine", F.weekofyear("Order_Date"))
    df = df.withColumn("Jour_Semaine", F.dayofweek("Order_Date"))

    df = df.withColumn("Delai_Livraison", F.datediff(F.col("Ship_Date"), F.col("Order_Date")))

    nb_final = df.count()
    if nb_final == 0:
        raise ValueError("Le DataFrame est vide apres nettoyage.")

    logger.info(f"   {nb_final:,} lignes apres nettoyage ({nb_initial - nb_final} supprimees)")
    return df


def get_dataset_stats(df) -> dict:
    """
    Statistiques descriptives.

    Nb_Commandes = countDistinct(Order_ID), pas count(*).
    Panier_Moyen = SUM(Sales) / Nb_Commandes (par commande, pas par ligne).
    """
    stats = {}
    stats["nb_lignes"] = df.count()
    stats["nb_colonnes"] = len(df.columns)

    date_stats = df.agg(
        F.min("Order_Date").alias("date_min"),
        F.max("Order_Date").alias("date_max"),
    ).collect()[0]
    stats["date_min"] = date_stats["date_min"]
    stats["date_max"] = date_stats["date_max"]

    stats["regions"] = [r[0] for r in df.select("Region").distinct().collect()]
    stats["categories"] = [r[0] for r in df.select("Category").distinct().collect()]
    stats["segments"] = [r[0] for r in df.select("Segment").distinct().collect()]
    stats["annees"] = sorted([r[0] for r in df.select("Annee").distinct().collect()])

    sales_stats = df.agg(
        F.round(F.sum("Sales"), 2).alias("ca_total"),
        F.round(F.min("Sales"), 2).alias("vente_min"),
        F.round(F.max("Sales"), 2).alias("vente_max"),
        F.countDistinct("Order_ID").alias("nb_commandes"),
        F.count("*").alias("nb_articles"),
    ).collect()[0]
    stats["ca_total"] = float(sales_stats["ca_total"])
    stats["vente_min"] = float(sales_stats["vente_min"])
    stats["vente_max"] = float(sales_stats["vente_max"])
    stats["nb_commandes"] = int(sales_stats["nb_commandes"])
    stats["nb_articles"] = int(sales_stats["nb_articles"])
    stats["panier_moyen"] = (
        round(stats["ca_total"] / stats["nb_commandes"], 2)
        if stats["nb_commandes"] else 0.0
    )

    stats["nb_clients"] = df.select("Customer_ID").distinct().count()
    stats["nb_produits"] = df.select("Product_ID").distinct().count()
    return stats


def afficher_apercu(df):
    """Imprime un apercu detaille du dataset."""
    stats = get_dataset_stats(df)

    print("\n" + "=" * 65)
    print("   DATASET CHARGE")
    print("=" * 65)

    print("\n  Structure")
    print(f"     Lignes      : {stats['nb_lignes']:,}")
    print(f"     Colonnes    : {stats['nb_colonnes']}")

    print("\n  Periode")
    print(f"     Debut       : {stats['date_min']}")
    print(f"     Fin         : {stats['date_max']}")
    print(f"     Annees      : {stats['annees']}")

    print("\n  Dimensions")
    print(f"     Regions     : {sorted(stats['regions'])}")
    print(f"     Categories  : {sorted(stats['categories'])}")
    print(f"     Segments    : {sorted(stats['segments'])}")

    print("\n  Metriques financieres")
    print(f"     CA Total    : {stats['ca_total']:,.2f} USD")
    print(f"     Panier moyen: {stats['panier_moyen']:,.2f} USD")
    print(f"     Vente min   : {stats['vente_min']:,.2f} USD")
    print(f"     Vente max   : {stats['vente_max']:,.2f} USD")

    print("\n  Volumetrie")
    print(f"     Commandes   : {stats['nb_commandes']:,}")
    print(f"     Clients     : {stats['nb_clients']:,}")
    print(f"     Produits    : {stats['nb_produits']:,}")

    print("\n  Apercu des donnees:")
    df.select(
        "Order_Date", "Region", "Category",
        "Sub_Category", "Sales", "Annee", "Trimestre",
    ).show(5, truncate=False)

    print("=" * 65)
    return stats


if __name__ == "__main__":
    try:
        spark = create_spark_session()
        df = load_data(spark)
        afficher_apercu(df)
        spark.stop()
        print("\n[OK] Etape 1 - load_data.py terminee")
    except Exception as e:
        logger.error(f"[KO] Erreur: {e}")
        raise
