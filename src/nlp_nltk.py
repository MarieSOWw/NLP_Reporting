"""
nlp_nltk.py - Traitement NLP avec NLTK
========================================

Role
----
Troisieme maillon du pipeline. Deux responsabilites distinctes :

1. EXTRACTION DE FAITS depuis les KPIs Spark (extraire_faits).
   Produit un dict structure qui sera ensuite passe au LLM Mistral
   pour generer le rapport narratif.

2. ANALYSE DU RAPPORT GENERE par Mistral (analyser_rapport_genere).
   Tokenisation, lemmatisation, NER, POS tagging, classification par
   intention business (Chantier 2), tonalite business par lexique
   metier (alternative a VADER pour le B2B et le francais).

Pourquoi NLTK et pas spaCy
--------------------------
- NLTK couvre l'ensemble : tokenisation, stopwords, POS, NER, VADER.
- spaCy serait plus rapide mais le but pedagogique est de valoriser la
  stack NLP classique au programme.

VADER (sentiment)
-----------------
VADER est anglais-only et calibrer pour les reseaux sociaux. Il score
"forte chute" comme positif a cause de "forte". Pour le francais et le
B2B on l'a remplace par evaluer_tonalite_business() qui s'appuie sur
un lexique metier dedie (croissance / decroissance / opportunite /
risque / volatilite). VADER reste utilise en anglais pour
retrocompatibilite.

Chantier 2 : Classification d'intentions business
--------------------------------------------------
Chaque phrase d'un rapport est tagguee par intention :
    tendance / anomalie / opportunite / risque / recommandation / contexte
Permet de mesurer la COUVERTURE BUSINESS du texte (est-ce qu'il parle
bien de risques ? de recommandations ?). Score sur 4 criteres pour
une note /100.

Tonalite business (finition Chantier 4)
---------------------------------------
Score borne [-100, +100] selon le ratio signaux positifs vs negatifs
dans le lexique metier. 3 niveaux : favorable / neutre / defavorable.

Lexiques
--------
- LEXIQUE_INTENTION_FR / EN : pour la classification (Chantier 2)
- LEXIQUE_THEMES_FR / EN    : pour la detection des themes dominants
  ET le calcul de la tonalite business (memes mots, deux usages)

Fonctions publiques principales
-------------------------------
- NLTKProcessor(langue)              : objet outillage NLP
- extraire_faits(kpis)               : faits structures pour Mistral
- analyser_rapport_genere(texte, lg) : analyse du rapport genere
- classifier_phrases_business(t, lg) : classification par intention
- mesurer_couverture_business(c)     : score de couverture /100
- extraire_themes_business(t, lg, n) : top N themes dominants
- evaluer_tonalite_business(t, lg)   : score business borne [-100, +100]
"""

import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if sys.platform == "win32":
    os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME", r"C:\hadoop")

import nltk

NLTK_DATA_DIR = PROJECT_ROOT / ".cache" / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.insert(0, str(NLTK_DATA_DIR))

NLTK_PACKAGES = [
    "punkt", "punkt_tab", "stopwords", "wordnet",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab", "words", "vader_lexicon",
]


def download_nltk_packages():
    for package in NLTK_PACKAGES:
        try:
            nltk.download(package, quiet=True, download_dir=str(NLTK_DATA_DIR))
        except Exception:
            pass


download_nltk_packages()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.load_data import create_spark_session, load_data
from src.analytics import compute_kpis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


LEXIQUE_INTENTION_FR = {
    "tendance": [
        "progresse", "progression", "augmente", "augmentation", "hausse",
        "croissance", "croit", "evolue", "evolution", "monte",
        "passe de", "atteint", "s'eleve", "se hisse", "trajectoire",
        "dynamique", "expansion", "essor",
    ],
    "anomalie": [
        "chute", "chuter", "recul", "recule", "baisse", "diminution",
        "effondrement", "effondre", "plonge", "plongeon", "decroche",
        "degringole", "devisse", "rupture", "anormal", "atypique",
        "creux", "point bas", "perd",
    ],
    "opportunite": [
        "potentiel", "opportunite", "capitaliser", "saisir", "exploiter",
        "levier", "renforcer", "consolider", "developper", "accelerer",
        "amplifier", "ouvre la voie", "permet de",
    ],
    "risque": [
        "dependance", "concentration", "volatilite", "volatile", "risque",
        "fragile", "fragilite", "vulnerable", "vulnerabilite", "exposition",
        "menace", "incertitude", "instabilite", "saisonnalite", "cyclique",
        "sensible", "expose",
    ],
    "recommandation": [
        "il convient", "il faut", "recommandons", "recommandation",
        "devrait", "doit", "suggerons", "preconisons", "proposons",
        "priorite", "prioritaire", "il serait pertinent", "il est necessaire",
        "mettre en place", "envisager", "planifier", "lancer",
    ],
}

LEXIQUE_INTENTION_EN = {
    "tendance": [
        "grows", "grew", "growing", "growth", "rises", "rose", "rising",
        "increases", "increased", "trending", "trend", "expands", "expansion",
        "climbs", "climbed", "reaches", "reached", "trajectory", "momentum",
    ],
    "anomalie": [
        "drop", "drops", "dropped", "fall", "fell", "falling", "decline",
        "declined", "plunge", "plunged", "collapse", "collapsed", "crash",
        "downturn", "anomaly", "abnormal", "trough", "low point", "loses",
    ],
    "opportunite": [
        "potential", "opportunity", "leverage", "capitalize", "capitalise",
        "seize", "exploit", "strengthen", "consolidate", "develop",
        "accelerate", "amplify", "enables", "allows",
    ],
    "risque": [
        "dependence", "dependency", "concentration", "volatility", "volatile",
        "risk", "fragile", "vulnerable", "vulnerability", "exposure",
        "threat", "uncertainty", "instability", "seasonality", "cyclical",
        "sensitive", "exposed",
    ],
    "recommandation": [
        "should", "must", "recommend", "recommendation", "suggest",
        "advise", "propose", "priority", "prioritize", "prioritise",
        "would be wise", "it is necessary", "consider", "plan", "launch",
        "implement",
    ],
}

LEXIQUE_THEMES_FR = {
    "croissance": [
        "croissance", "progression", "augmentation", "hausse", "expansion",
        "dynamique positive", "essor", "acceleration", "performance record",
        "hausse exceptionnelle", "meilleure performance",
    ],
    "decroissance": [
        "chute", "baisse", "recul", "declin", "diminution", "repli",
        "contraction", "effondrement", "ralentissement", "degradation",
        "creux", "pire performance",
    ],
    "saisonnalite": [
        "saisonnalite", "saisonnier", "cyclique", "trimestriel", "recurrent",
        "pic de fin d'annee", "pattern saisonnier", "cycle",
        "debut d'annee", "fin d'annee", "t1", "t2", "t3", "t4",
    ],
    "concentration_produit": [
        "sous-categorie", "sous-categories", "produit phare", "phones",
        "chairs", "storage", "concentration", "dependance produit",
        "portefeuille produit",
    ],
    "concentration_regionale": [
        "region", "west", "east", "south", "central", "regional",
        "zone geographique", "marche regional", "desequilibre regional",
    ],
    "volatilite": [
        "volatilite", "volatile", "instabilite", "variations", "fluctuation",
        "ecarts", "irreguliere", "imprevisible", "turbulence",
    ],
    "fidelisation": [
        "client", "clients", "segment", "consumer", "corporate", "home office",
        "fidelisation", "base client", "retention", "lifetime value",
    ],
    "panier_moyen": [
        "panier moyen", "ticket moyen", "valeur moyenne", "upselling",
        "cross-selling", "aov",
    ],
    "diversification": [
        "diversification", "diversifier", "elargir", "complementaire",
        "offre complementaire", "nouveaux segments", "expansion produit",
    ],
    "risque": [
        "risque", "fragilite", "vulnerabilite", "menace", "exposition",
        "alerte", "critique", "inquietant", "preoccupant",
    ],
    "opportunite": [
        "opportunite", "levier", "potentiel", "capitaliser", "exploiter",
        "renforcer", "amplifier", "accelerer", "saisir", "upside",
    ],
    "recommandation": [
        "recommandation", "action", "mesure", "plan", "strategie",
        "il est conseille", "il convient", "nous recommandons", "doit",
        "mettre en place", "lancer", "investir",
    ],
}

LEXIQUE_THEMES_EN = {
    "croissance": [
        "growth", "increase", "rise", "expansion", "uptrend",
        "surge", "boom", "record performance", "strong momentum",
        "exceptional growth", "best performance",
    ],
    "decroissance": [
        "drop", "decline", "decrease", "downturn", "fall", "contraction",
        "slowdown", "deterioration", "weakness", "trough", "worst performance",
    ],
    "saisonnalite": [
        "seasonality", "seasonal", "cyclical", "quarterly", "recurring",
        "year-end peak", "seasonal pattern", "cycle",
        "early year", "end of year", "q1", "q2", "q3", "q4",
    ],
    "concentration_produit": [
        "sub-category", "subcategory", "top product", "phones",
        "chairs", "storage", "concentration", "product dependence",
        "product portfolio",
    ],
    "concentration_regionale": [
        "region", "west", "east", "south", "central", "regional",
        "geographic zone", "regional market", "regional imbalance",
    ],
    "volatilite": [
        "volatility", "volatile", "instability", "variations", "fluctuation",
        "swings", "irregular", "unpredictable", "turbulence",
    ],
    "fidelisation": [
        "customer", "customers", "segment", "consumer", "corporate",
        "home office", "retention", "loyalty", "customer base", "lifetime value",
    ],
    "panier_moyen": [
        "average basket", "basket size", "average ticket", "upselling",
        "cross-selling", "aov",
    ],
    "diversification": [
        "diversification", "diversify", "broaden", "complementary",
        "complementary offer", "new segments", "product expansion",
    ],
    "risque": [
        "risk", "fragility", "vulnerability", "threat", "exposure",
        "alert", "critical", "concerning", "worrying",
    ],
    "opportunite": [
        "opportunity", "lever", "potential", "capitalize", "leverage",
        "reinforce", "amplify", "accelerate", "upside",
    ],
    "recommandation": [
        "recommendation", "action", "plan", "strategy",
        "we recommend", "should", "must", "launch", "invest",
        "set up", "implement",
    ],
}


class NLTKProcessor:
    """
    Outillage NLP : tokenisation, lemmatisation, stopwords, POS, NER,
    sentiment VADER (anglais uniquement), extraction de mots-cles.

    VADER est desactive si langue != 'english'. Pour le francais,
    utiliser evaluer_tonalite_business().
    """

    def __init__(self, langue: str = "english"):
        self.langue = langue
        try:
            self.stopwords = set(stopwords.words(langue))
        except Exception:
            self.stopwords = set()
            logger.warning(f"Stopwords non disponibles pour '{langue}'")

        self.stopwords.update({
            "usd", "dollar", "dollars", "year", "years", "quarter",
            "q1", "q2", "q3", "q4", "region", "total", "sales",
            "revenue", "growth", "increase", "decrease", "percent",
        })

        self.lemmatizer = WordNetLemmatizer()

        if langue == "english":
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self._vader_active = True
            except Exception:
                self.sentiment_analyzer = None
                self._vader_active = False
                logger.warning("VADER non disponible")
        else:
            self.sentiment_analyzer = None
            self._vader_active = False
            logger.info(
                f"NLTKProcessor : VADER desactive pour langue='{langue}' "
                f"(VADER est anglais-only). Utiliser evaluer_tonalite_business()."
            )

        logger.info(f"[OK] NLTKProcessor initialise (langue: {langue}, vader={self._vader_active})")

    def tokenize_sentences(self, text: str) -> list:
        try:
            return sent_tokenize(text)
        except Exception:
            return text.split(".")

    def tokenize_words(self, text: str) -> list:
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t.lower() not in self.stopwords]

    def lemmatize(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(t.lower()) for t in tokens]

    def preprocess(self, text: str) -> list:
        """Tokenize -> filtre alpha -> remove stopwords -> lemmatize."""
        tokens = self.tokenize_words(text)
        tokens = [t for t in tokens if t.isalpha()]
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

    def pos_tagging(self, text: str) -> list:
        tokens = self.tokenize_words(text)
        try:
            return pos_tag(tokens)
        except Exception:
            return [(t, "NN") for t in tokens]

    def extract_entities(self, text: str) -> dict:
        """NER + regex MONEY / PERCENT."""
        entities = {
            "PERSON": [], "ORGANIZATION": [], "GPE": [],
            "MONEY": [], "PERCENT": [], "DATE": [],
        }
        try:
            tokens = self.tokenize_words(text)
            tagged = pos_tag(tokens)
            chunked = ne_chunk(tagged)
            for subtree in chunked:
                if isinstance(subtree, Tree):
                    entity_type = subtree.label()
                    entity_value = " ".join([tok for tok, _ in subtree.leaves()])
                    if entity_type in entities:
                        entities[entity_type].append(entity_value)
        except Exception as e:
            logger.warning(f"Erreur NER: {e}")

        money_pattern = r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?)'
        percent_pattern = r"[+-]?\d+(?:\.\d+)?%"
        entities["MONEY"].extend(re.findall(money_pattern, text, re.IGNORECASE))
        entities["PERCENT"].extend(re.findall(percent_pattern, text))

        for key in entities:
            entities[key] = list(set(entities[key]))
        return entities

    def analyze_sentiment(self, text: str) -> dict:
        """Renvoie un dict VADER. Scores neutres si VADER desactive."""
        if not self._vader_active or self.sentiment_analyzer is None:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        try:
            return self.sentiment_analyzer.polarity_scores(text)
        except Exception:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    def extract_keywords(self, text: str, top_n: int = 10) -> list:
        tokens = self.preprocess(text)
        counter = Counter(tokens)
        return counter.most_common(top_n)

    def generate_summary_stats(self, text: str) -> dict:
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)
        return {
            "nb_phrases": len(sentences),
            "nb_mots": len(words),
            "mots_uniques": len(set(words)),
            "longueur_moyenne_phrase": len(words) / max(len(sentences), 1),
        }


# Chantier 2 : classification d'intentions business

def _detecter_intention_phrase(phrase: str, langue: str = "fr") -> str:
    """Detecte l'intention business d'une phrase via lexique de regles."""
    lex = LEXIQUE_INTENTION_FR if langue == "fr" else LEXIQUE_INTENTION_EN
    phrase_low = phrase.lower()
    scores = {cat: 0 for cat in lex.keys()}
    for cat, mots in lex.items():
        for mot in mots:
            if mot in phrase_low:
                scores[cat] += 1
    if max(scores.values()) == 0:
        return "contexte"
    priorites = ["recommandation", "anomalie", "risque", "opportunite", "tendance"]
    for cat in priorites:
        if scores[cat] > 0:
            return cat
    return "contexte"


def classifier_phrases_business(texte: str, langue: str = "fr") -> dict:
    """
    Tag chaque phrase d'un texte par intention business.
    Renvoie {"phrases": [...], "compteurs": {...}, "total_phrases": N,
             "phrases_actionnables": N}.
    """
    if not texte or not texte.strip():
        return {
            "phrases": [],
            "compteurs": {k: 0 for k in [
                "tendance", "anomalie", "opportunite",
                "risque", "recommandation", "contexte",
            ]},
            "total_phrases": 0,
            "phrases_actionnables": 0,
        }

    try:
        phrases = sent_tokenize(texte)
    except Exception:
        phrases = [p.strip() for p in re.split(r"[.!?]+", texte) if p.strip()]

    classifications = []
    compteurs = {k: 0 for k in [
        "tendance", "anomalie", "opportunite",
        "risque", "recommandation", "contexte",
    ]}

    for phrase in phrases:
        phrase_clean = phrase.strip()
        if len(phrase_clean) < 10:
            continue
        intention = _detecter_intention_phrase(phrase_clean, langue=langue)
        classifications.append({"phrase": phrase_clean, "intention": intention})
        compteurs[intention] += 1

    phrases_actionnables = sum(v for k, v in compteurs.items() if k != "contexte")
    return {
        "phrases": classifications,
        "compteurs": compteurs,
        "total_phrases": len(classifications),
        "phrases_actionnables": phrases_actionnables,
    }


def mesurer_couverture_business(classifications: dict) -> dict:
    """
    Score /100 base sur 4 criteres : presence d'au moins 1 tendance,
    1 alerte (anomalie ou risque), 2 recommandations, 1 opportunite.
    """
    compteurs = classifications.get("compteurs", {})
    criteres = {
        "tendance_presente":    compteurs.get("tendance", 0) >= 1,
        "alerte_presente":      (compteurs.get("anomalie", 0) + compteurs.get("risque", 0)) >= 1,
        "deux_recommandations": compteurs.get("recommandation", 0) >= 2,
        "opportunite_presente": compteurs.get("opportunite", 0) >= 1,
    }
    score = sum(25 for v in criteres.values() if v)

    if score >= 90:
        mention = "Excellente couverture"
    elif score >= 70:
        mention = "Bonne couverture"
    elif score >= 50:
        mention = "Couverture correcte"
    elif score >= 25:
        mention = "Couverture partielle"
    else:
        mention = "Couverture insuffisante"

    lacunes = []
    if not criteres["tendance_presente"]:
        lacunes.append("Aucune tendance identifiee dans le rapport")
    if not criteres["alerte_presente"]:
        lacunes.append("Aucune anomalie ni risque signale")
    if not criteres["deux_recommandations"]:
        manque = 2 - compteurs.get("recommandation", 0)
        lacunes.append(f"Manque {manque} recommandation(s) concrete(s)")
    if not criteres["opportunite_presente"]:
        lacunes.append("Aucune opportunite explicitement formulee")

    return {
        "score_couverture": score,
        "criteres": criteres,
        "mention": mention,
        "lacunes": lacunes,
    }


def extraire_themes_business(texte: str, langue: str = "fr", top_n: int = 5) -> list:
    """Top N themes business dominants via lexique metier."""
    if not texte:
        return []
    lex = LEXIQUE_THEMES_FR if langue == "fr" else LEXIQUE_THEMES_EN
    texte_low = texte.lower()
    scores = {}
    for theme, mots in lex.items():
        score = 0
        for mot in mots:
            score += texte_low.count(mot)
        if score > 0:
            scores[theme] = score
    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]


def evaluer_tonalite_business(texte: str, langue: str = "fr") -> dict:
    """
    Tonalite business : score borne [-100, +100] selon ratio
    signaux positifs vs negatifs. Niveau de risque et stabilite calcules
    a partir des memes lexiques.
    """
    lex = LEXIQUE_THEMES_FR if langue == "fr" else LEXIQUE_THEMES_EN
    texte_low = texte.lower() if texte else ""

    pos = sum(texte_low.count(m) for m in lex.get("croissance", []))
    pos += sum(texte_low.count(m) for m in lex.get("opportunite", []))

    neg = sum(texte_low.count(m) for m in lex.get("decroissance", []))
    neg += sum(texte_low.count(m) for m in lex.get("risque", []))

    risque = sum(texte_low.count(m) for m in lex.get("risque", []))
    risque += sum(texte_low.count(m) for m in lex.get("volatilite", []))

    instabilite = sum(texte_low.count(m) for m in lex.get("volatilite", []))
    instabilite += sum(texte_low.count(m) for m in lex.get("saisonnalite", []))

    total_signaux = pos + neg
    score_business = int(((pos - neg) / total_signaux) * 100) if total_signaux > 0 else 0

    if score_business > 20:
        tonalite = "favorable"
        label_fr = "Favorable"
        label_en = "Favorable"
    elif score_business < -20:
        tonalite = "defavorable"
        label_fr = "Defavorable"
        label_en = "Unfavorable"
    else:
        tonalite = "neutre"
        label_fr = "Neutre"
        label_en = "Neutral"

    if risque >= 5:
        niveau_risque = "eleve"
        risque_fr = "Eleve"
        risque_en = "High"
    elif risque >= 2:
        niveau_risque = "modere"
        risque_fr = "Modere"
        risque_en = "Moderate"
    else:
        niveau_risque = "faible"
        risque_fr = "Faible"
        risque_en = "Low"

    stabilite = "instable" if instabilite >= 3 else "stable"
    stab_fr = "Instable" if stabilite == "instable" else "Stable"
    stab_en = "Unstable" if stabilite == "instable" else "Stable"

    return {
        "tonalite": tonalite,
        "tonalite_label": label_fr if langue == "fr" else label_en,
        "niveau_risque": niveau_risque,
        "niveau_risque_label": risque_fr if langue == "fr" else risque_en,
        "stabilite": stabilite,
        "stabilite_label": stab_fr if langue == "fr" else stab_en,
        "score_business": score_business,
        "signaux_positifs": pos,
        "signaux_negatifs": neg,
        "signaux_risque": risque,
    }


def extraire_faits(kpis: dict) -> dict:
    """
    Structure tous les faits cles a partir des KPIs Spark.
    Ce dict est consomme par nlp_transformers.generer_rapport().
    """
    logger.info("Extraction des faits...")
    faits = {}

    annuel = kpis["annuel"].collect() if hasattr(kpis["annuel"], "collect") else kpis["annuel"]
    faits["annuel"] = sorted([
        {
            "annee": int(r["Annee"]),
            "ca": float(r["CA_Annuel"]),
            "commandes": int(r["Nb_Commandes"]),
            "panier_moyen": float(r["Panier_Moyen"]),
            "croissance_yoy": float(r["Croissance_YoY"]) if r["Croissance_YoY"] else None,
        }
        for r in annuel
    ], key=lambda x: x["annee"])

    faits["meilleure_annee"] = max(faits["annuel"], key=lambda x: x["ca"])
    faits["pire_annee"] = min(faits["annuel"], key=lambda x: x["ca"])

    annees_triees = faits["annuel"]
    ca_debut = annees_triees[0]["ca"]
    ca_fin = annees_triees[-1]["ca"]
    annee_debut = annees_triees[0]["annee"]
    annee_fin = annees_triees[-1]["annee"]

    faits["croissance_globale"] = round((ca_fin - ca_debut) / ca_debut * 100, 1) if ca_debut else 0
    faits["periode"] = f"{annee_debut}-{annee_fin}"

    variations = kpis["variation"].collect() if hasattr(kpis["variation"], "collect") else kpis["variation"]
    faits["variations"] = []
    for r in variations:
        if r["Annee"] and r["Variation_Pct"] is not None:
            faits["variations"].append({
                "annee": int(r["Annee"]),
                "trimestre": int(r["Trimestre"]),
                "ventes": float(r["Ventes_Totales"]),
                "variation": float(r["Variation_Pct"]),
            })

    if faits["variations"]:
        faits["meilleur_trim"] = max(faits["variations"], key=lambda x: x["variation"])
        faits["pire_trim"] = min(faits["variations"], key=lambda x: x["variation"])
    else:
        faits["meilleur_trim"] = {
            "annee": annee_fin, "trimestre": 1, "ventes": ca_fin, "variation": 0.0,
        }
        faits["pire_trim"] = faits["meilleur_trim"]

    top = kpis["top_produits"].collect() if hasattr(kpis["top_produits"], "collect") else kpis["top_produits"]
    faits["top_produits"] = [
        {"nom": r["Sub_Category"], "ventes": float(r["Ventes_Totales"]), "qte": int(r["Quantite_Vendue"])}
        for r in top[:3]
    ]

    regions = kpis["meilleure_region"].collect() if hasattr(kpis["meilleure_region"], "collect") else kpis["meilleure_region"]
    faits["regions"] = {}
    annees_vues = []
    for r in regions:
        if r["Annee"] not in annees_vues:
            annees_vues.append(r["Annee"])
            faits["regions"][int(r["Annee"])] = {
                "region": r["Region"],
                "ventes": float(r["Ventes_Totales"]),
            }

    mensuel = kpis["mensuel"].collect() if hasattr(kpis["mensuel"], "collect") else kpis["mensuel"]
    faits["mensuel"] = [
        {
            "annee": int(r["Annee"]), "mois": int(r["Mois"]),
            "ventes": float(r["Ventes_Mensuelles"]),
            "commandes": int(r["Nb_Commandes"]),
        }
        for r in mensuel
    ]

    segments = kpis["segment"].collect() if hasattr(kpis["segment"], "collect") else kpis["segment"]
    faits["segments"] = {}
    for r in segments:
        seg = r["Segment"]
        if seg not in faits["segments"]:
            faits["segments"][seg] = {"ventes_total": 0.0, "commandes": 0}
        faits["segments"][seg]["ventes_total"] += float(r["Ventes_Segment"])
        faits["segments"][seg]["commandes"] += int(r["Nb_Commandes"])

    faits["tendance_globale"] = _categoriser_tendance(faits["croissance_globale"])
    faits["volatilite"] = _calculer_volatilite(faits["variations"])

    logger.info("[OK] Faits extraits")
    return faits


def analyser_rapport_genere(rapport_texte: str, langue: str = "fr") -> dict:
    """
    Analyse NLTK du texte du rapport Mistral. Format de retour aplati
    et compatible avec save_to_postgres.sauvegarder_nltk_analysis.
    """
    if not rapport_texte:
        return {
            "sentiment": {"neg": 0, "neu": 1, "pos": 0, "compound": 0},
            "keywords": [],
            "nb_phrases": 0,
            "nb_mots": 0,
            "stats": {"nb_phrases": 0, "nb_mots": 0, "mots_uniques": 0, "longueur_moyenne_phrase": 0},
            "classifications": [],
            "couverture": {"score": 0, "mention": "Vide", "lacunes": []},
            "themes": [],
            "tonalite": {"label": "neutre", "valeur": 0},
            "tonalite_business": {},
        }

    nltk_lang = "english" if langue == "en" else "french"
    nlp = NLTKProcessor(langue=nltk_lang)

    stats = nlp.generate_summary_stats(rapport_texte)
    sentiment = nlp.analyze_sentiment(rapport_texte)
    keywords = nlp.extract_keywords(rapport_texte, top_n=10)

    classifications_dict = classifier_phrases_business(rapport_texte, langue=langue)
    couverture_dict = mesurer_couverture_business(classifications_dict)
    themes_tuples = extraire_themes_business(rapport_texte, langue=langue, top_n=5)
    tonalite_dict = evaluer_tonalite_business(rapport_texte, langue=langue)

    classifications_flat = [
        {"intent": p["intention"], "sentence": p["phrase"]}
        for p in classifications_dict.get("phrases", [])
    ]
    couverture_flat = {
        "score": couverture_dict.get("score_couverture", 0),
        "mention": couverture_dict.get("mention", ""),
        "lacunes": couverture_dict.get("lacunes", []),
        "criteres": couverture_dict.get("criteres", {}),
    }
    themes_flat = [theme for theme, _score in themes_tuples]
    tonalite_flat = {
        "label": tonalite_dict.get("tonalite", "neutre"),
        "label_lisible": tonalite_dict.get("tonalite_label", ""),
        "valeur": tonalite_dict.get("score_business", 0),
        "niveau_risque": tonalite_dict.get("niveau_risque", "faible"),
        "stabilite": tonalite_dict.get("stabilite", "stable"),
    }

    return {
        "sentiment": sentiment,
        "keywords": keywords,
        "nb_phrases": stats["nb_phrases"],
        "nb_mots": stats["nb_mots"],
        "stats": stats,
        "classifications": classifications_flat,
        "couverture": couverture_flat,
        "themes": themes_flat,
        "tonalite": tonalite_flat,
        "tonalite_business": tonalite_dict,
        "compteurs": classifications_dict.get("compteurs", {}),
    }


def _generer_texte_brut(faits: dict) -> str:
    """Texte de demonstration (illustratif uniquement, pas utilise en production)."""
    return (
        f"Business performance report for Superstore covering the period {faits['periode']}.\n"
        f"Total revenue grew by {faits['croissance_globale']}% over this period.\n"
        f"The best year was {faits['meilleure_annee']['annee']} with ${faits['meilleure_annee']['ca']:,.0f} in revenue.\n"
        f"The strongest quarter was Q{faits['meilleur_trim']['trimestre']} {faits['meilleur_trim']['annee']}\n"
        f"with a growth of {faits['meilleur_trim']['variation']}%.\n"
        f"Top performing sub-categories include {', '.join([p['nom'] for p in faits['top_produits']])}.\n"
    )


def _categoriser_tendance(croissance: float) -> dict:
    if croissance is None:
        return {"label": "Indeterminee", "color": "gray"}
    if croissance >= 50:
        return {"label": "Croissance exceptionnelle", "color": "green"}
    elif croissance >= 20:
        return {"label": "Forte croissance", "color": "green"}
    elif croissance >= 5:
        return {"label": "Croissance moderee", "color": "blue"}
    elif croissance >= -5:
        return {"label": "Stable", "color": "gray"}
    elif croissance >= -20:
        return {"label": "Declin modere", "color": "orange"}
    else:
        return {"label": "Declin severe", "color": "red"}


def _calculer_volatilite(variations: list) -> dict:
    if not variations:
        return {"niveau": "Inconnue", "valeur": 0}
    vals = [abs(v["variation"]) for v in variations if v["variation"] is not None]
    if not vals:
        return {"niveau": "Inconnue", "valeur": 0}
    moyenne = sum(vals) / len(vals)
    if moyenne < 10:
        niveau = "Faible"
    elif moyenne < 20:
        niveau = "Moderee"
    elif moyenne < 30:
        niveau = "Elevee"
    else:
        niveau = "Tres elevee"
    return {"niveau": niveau, "valeur": round(moyenne, 1)}


def construire_prompt_t5(faits: dict, region: str = None, annee: int = None) -> str:
    """Prompt T5-style (conserve pour compatibilite ; non utilise par defaut)."""
    mt = faits["meilleur_trim"]
    pt = faits["pire_trim"]
    t3 = faits["top_produits"]
    ba = faits["meilleure_annee"]

    meilleur_segment = max(
        faits["segments"].items(),
        key=lambda x: x[1]["ventes_total"],
    ) if faits.get("segments") else ("N/A", {"ventes_total": 0})

    tendance = faits.get("tendance_globale", {}).get("label", "N/A")

    mt_var = mt.get("variation")
    pt_var = pt.get("variation")
    mt_var_str = f"{mt_var:+.1f}%" if mt_var is not None else "N/A"
    pt_var_str = f"{pt_var:+.1f}%" if pt_var is not None else "N/A"

    prompt = (
        f"summarize: Business performance report for Superstore "
        f"({faits['periode']}). "
        f"Overall trend: {tendance}. "
        f"Total revenue grew by {faits['croissance_globale']}% over the period. "
        f"Best year was {ba['annee']} with ${ba['ca']:,.0f} revenue and {ba['commandes']} orders. "
        f"Strongest quarter: Q{mt['trimestre']} {mt['annee']} with {mt_var_str} growth reaching ${mt['ventes']:,.0f} in sales. "
        f"Weakest quarter: Q{pt['trimestre']} {pt['annee']} with {pt_var_str} change. "
        f"Top 3 sub-categories: "
        f"{t3[0]['nom']} (${t3[0]['ventes']:,.0f}), "
        f"{t3[1]['nom']} (${t3[1]['ventes']:,.0f}), "
        f"{t3[2]['nom']} (${t3[2]['ventes']:,.0f}). "
        f"Best customer segment: {meilleur_segment[0]} (${meilleur_segment[1]['ventes_total']:,.0f}). "
        f"Yearly revenue: "
        + ", ".join([f"{r['annee']}: ${r['ca']:,.0f}" for r in faits["annuel"]]) + "."
    )
    return prompt


def afficher_faits(faits: dict):
    """Imprime les faits dans la console."""
    print("\n" + "=" * 65)
    print("   FAITS EXTRAITS")
    print("=" * 65)
    print(f"\n  Periode analysee      : {faits['periode']}")
    croiss = faits.get("croissance_globale")
    croiss_txt = f"{croiss:+.1f}%" if croiss is not None else "N/A"
    print(f"  Croissance globale    : {croiss_txt}")
    print(f"  Tendance              : {faits['tendance_globale']['label']}")
    print(f"  Volatilite            : {faits['volatilite']['niveau']} ({faits['volatilite']['valeur']}%)")
    print(f"\n  Meilleure annee       : {faits['meilleure_annee']['annee']} - {faits['meilleure_annee']['ca']:,.0f} USD")
    print(f"  Pire annee            : {faits['pire_annee']['annee']} - {faits['pire_annee']['ca']:,.0f} USD")
    mt_var = faits['meilleur_trim'].get('variation')
    pt_var = faits['pire_trim'].get('variation')
    mt_var_str = f"{mt_var:+.1f}%" if mt_var is not None else "N/A"
    pt_var_str = f"{pt_var:+.1f}%" if pt_var is not None else "N/A"
    print(f"\n  Meilleur trimestre    : T{faits['meilleur_trim']['trimestre']} {faits['meilleur_trim']['annee']} ({mt_var_str})")
    print(f"  Pire trimestre        : T{faits['pire_trim']['trimestre']} {faits['pire_trim']['annee']} ({pt_var_str})")
    print(f"\n  Top 3 sous-categories:")
    for i, p in enumerate(faits["top_produits"], 1):
        print(f"     {i}. {p['nom']:<20} {p['ventes']:>12,.0f} USD ({p['qte']} commandes)")
    print(f"\n  Meilleures regions par annee:")
    for annee, info in sorted(faits["regions"].items()):
        print(f"     {annee} -> {info['region']:<10} {info['ventes']:>10,.0f} USD")
    print(f"\n  Ventes par segment:")
    for seg, info in sorted(faits["segments"].items(), key=lambda x: -x[1]["ventes_total"]):
        print(f"     {seg:<20} {info['ventes_total']:>12,.0f} USD ({info['commandes']} commandes)")
    print("\n" + "=" * 65)


def afficher_classifications(classifications: dict, couverture: dict, themes: list, tonalite: dict):
    """Imprime les resultats du Chantier 2."""
    print("\n" + "=" * 65)
    print("   ANALYSE BUSINESS DU RAPPORT (Chantier 2)")
    print("=" * 65)
    print(f"\n  Compteurs par intention:")
    for cat, n in classifications["compteurs"].items():
        print(f"     {cat:<18} {n}")
    print(f"\n  Couverture business : {couverture['score_couverture']}/100 - {couverture['mention']}")
    if couverture["lacunes"]:
        print(f"  Lacunes detectees:")
        for l in couverture["lacunes"]:
            print(f"     - {l}")
    if themes:
        print(f"\n  Themes business dominants:")
        for theme, score in themes:
            print(f"     - {theme:<25} (score {score})")
    print(f"\n  Tonalite business:")
    print(f"     Tonalite       : {tonalite['tonalite']}")
    print(f"     Niveau risque  : {tonalite['niveau_risque']}")
    print(f"     Stabilite      : {tonalite['stabilite']}")
    print(f"     Score business : {tonalite['score_business']:+d}")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    try:
        spark = create_spark_session()
        df = load_data(spark)
        kpis = compute_kpis(df)
        faits = extraire_faits(kpis)
        afficher_faits(faits)

        prompt = construire_prompt_t5(faits)
        print(f"\n  Prompt pour Mistral:")
        print(f"  {prompt[:500]}...")
        print(f"  [Prompt complet : {len(prompt)} caracteres]")

        texte_demo = _generer_texte_brut(faits)
        classif = classifier_phrases_business(texte_demo, langue="en")
        couv = mesurer_couverture_business(classif)
        themes = extraire_themes_business(texte_demo, langue="en")
        ton = evaluer_tonalite_business(texte_demo, langue="en")
        afficher_classifications(classif, couv, themes, ton)

        spark.stop()
        print("\n[OK] Etape 3 - nlp_nltk.py terminee")
    except Exception as e:
        logger.error(f"[KO] Erreur: {e}")
        raise
