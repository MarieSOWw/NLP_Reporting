"""
nlp_transformers.py - Generation NLP avec Mistral API
=======================================================

Role
----
Quatrieme maillon du pipeline. Prend le dict de faits produit par
nlp_nltk.extraire_faits() et le transforme en rapport business narratif
fluide en francais ou anglais.

Architecture en 2 etapes (Chantier 1)
--------------------------------------
1. generer_structure_json(faits, langue) :
   Demande a Mistral un JSON STRICTEMENT structure avec :
     - resume, tendance, meilleure et pire periodes
     - top produits, segment leader
     - risques, opportunites
     - recommandations hierarchisees (priorite + niveau + confiance + justification)
   Le JSON est valide contre les faits reels avant utilisation.

2. _generer_texte_depuis_structure(structure, faits, langue) :
   Demande a Mistral de transformer ce JSON en 4 paragraphes de prose
   (~400-600 mots) sans markdown ni structure visuelle.

Pourquoi ce pipeline en 2 etapes
--------------------------------
- Anti-hallucination : le LLM produit d'abord un JSON valide CONTRE
  les faits reels, puis habille ce JSON en prose. Il ne re-calcule
  jamais de chiffres dans la prose.
- Si le LLM echoue, on tombe sur un fallback prose qui utilise aussi
  le JSON (donc coherent avec les recommandations hierarchisees).

Detection d'anomalies (YoY +/- 20%)
-----------------------------------
detecter_anomalies() compare meme trimestre annee precedente. Seuil
canonique du projet : +/- 20% (aligne config.ANOMALY_THRESHOLD,
db.get_anomalies, system prompts du chatbot, mentions PDF).

Score de performance (calculer_score)
-------------------------------------
Score /100 base sur 4 criteres (poids configurables dans
config.SCORE_WEIGHTS) :
- croissance globale     (30 pts)
- regularite trimestrielle (25 pts)
- meilleur trimestre     (25 pts)
- diversite produits     (20 pts)

Generation hierarchisee des recommandations (Chantier 4)
---------------------------------------------------------
Chaque recommandation porte 3 attributs :
  priorite   : haute / moyenne / basse
  niveau     : strategique / tactique / operationnel
  confiance  : elevee / moyenne / faible (derivee de la volatilite)
+ une justification chiffree obligatoire.
Triees automatiquement par priorite avant rendu PDF.

Fonctions publiques
-------------------
- generer_structure_json(faits, langue)
- valider_structure_json(structure, faits, langue)
- generer_rapport(faits, langue)
- generer_rapport_avec_structure(faits, langue) -> {"rapport", "structure"}
- generer_resume_executif(faits, langue)
- detecter_anomalies(faits, seuil_alerte=20)
- calculer_score(faits)
- generer_tout(faits, langue) -> pipeline complet (rapport + structure + score)
"""

import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if sys.platform == "win32":
    os.environ["HADOOP_HOME"] = os.getenv("HADOOP_HOME", r"C:\hadoop")

warnings.filterwarnings("ignore")

from src.text_utils import strip_markdown as _strip_markdown
from src.load_data import create_spark_session, load_data
from src.analytics import compute_kpis
from src.nlp_nltk import extraire_faits, construire_prompt_t5, analyser_rapport_genere

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from config import MISTRAL_API_KEY, MISTRAL_MODEL, ANOMALY_THRESHOLD, SCORE_WEIGHTS
except ImportError:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    ANOMALY_THRESHOLD = 20.0
    SCORE_WEIGHTS = {"croissance": 30, "regularite": 25, "meilleur_trim": 25, "diversite": 20}


def _construire_prompt_json(faits: dict, langue: str) -> tuple:
    """Construit (system_msg, user_msg) pour la generation JSON."""
    ba = faits["meilleure_annee"]
    pa = faits.get("pire_annee", ba)
    mt = faits["meilleur_trim"]
    pt = faits["pire_trim"]
    t3 = faits["top_produits"]
    vol = faits.get("volatilite", {})
    tendance = faits.get("tendance_globale", {}).get("label", "N/A")

    region_focus = faits.get("region_focus")
    categorie_focus = faits.get("categorie_focus")
    nb_annees = len({a["annee"] for a in faits.get("annuel", [])})
    annee_unique = nb_annees == 1

    perimetre_lines_fr, perimetre_lines_en = [], []
    if region_focus:
        perimetre_lines_fr.append(f"- Perimetre regional : {region_focus} UNIQUEMENT")
        perimetre_lines_en.append(f"- Regional scope: {region_focus} ONLY")
    if categorie_focus:
        perimetre_lines_fr.append(f"- Perimetre categoriel : {categorie_focus} UNIQUEMENT")
        perimetre_lines_en.append(f"- Category scope: {categorie_focus} ONLY")
    if annee_unique:
        annee_uniq = faits["annuel"][0]["annee"]
        perimetre_lines_fr.append(f"- Perimetre temporel : annee {annee_uniq} UNIQUEMENT")
        perimetre_lines_en.append(f"- Temporal scope: year {annee_uniq} ONLY")

    perimetre_block_fr = ""
    perimetre_block_en = ""
    if perimetre_lines_fr:
        perimetre_block_fr = (
            "\nPERIMETRE FILTRE - IMPORTANT :\n"
            + "\n".join(perimetre_lines_fr)
            + "\nN'ecris JAMAIS 'L'annee X est le meilleur exercice' ni 'la region Y domine' "
              "sur les dimensions filtrees : c'est trivialement vrai puisque c'est le seul "
              "element du perimetre. Concentre-toi plutot sur les VRAIES insights : "
              "trimestres, sous-categories, segments, panier moyen, volatilite.\n"
        )
        perimetre_block_en = (
            "\nFILTERED SCOPE - IMPORTANT:\n"
            + "\n".join(perimetre_lines_en)
            + "\nNEVER write 'Year X is the best year' or 'Region Y dominates' on filtered "
              "dimensions: it's trivially true. Focus instead on REAL insights: quarters, "
              "sub-categories, segments, average basket, volatility.\n"
        )

    meilleur_segment = max(
        faits["segments"].items(),
        key=lambda x: x[1]["ventes_total"],
    ) if faits.get("segments") else ("N/A", {"ventes_total": 0})

    annuel_str = ", ".join([f"{r['annee']}={r['ca']:,.0f}USD" for r in faits["annuel"]])

    vol_niveau = vol.get("niveau", "Moderee").lower()
    if "faible" in vol_niveau:
        confiance_globale = "elevee"
    elif "elevee" in vol_niveau or "tres" in vol_niveau:
        confiance_globale = "faible"
    else:
        confiance_globale = "moyenne"

    cg_raw = faits.get("croissance_globale")
    if cg_raw is None:
        cg_str_fr = "non calculable (pas d'annee de reference comparable)"
        cg_str_en = "not computable (no comparable reference year)"
    else:
        cg_str_fr = f"{float(cg_raw):+.1f}%"
        cg_str_en = f"{float(cg_raw):+.1f}%"

    mt_var = mt.get("variation")
    pt_var = pt.get("variation")
    mt_var_fr = f"{mt_var:+.1f}%" if mt_var is not None else "QoQ N/A (premier trimestre du perimetre)"
    pt_var_fr = f"{pt_var:+.1f}%" if pt_var is not None else "QoQ N/A (premier trimestre du perimetre)"
    mt_var_en = f"{mt_var:+.1f}%" if mt_var is not None else "QoQ N/A (first quarter)"
    pt_var_en = f"{pt_var:+.1f}%" if pt_var is not None else "QoQ N/A (first quarter)"

    schema_fr = """{
  "summary": "Phrase de resume global en francais (1-2 phrases)",
  "main_trend": "Description de la tendance macro",
  "best_period": "Description de la meilleure periode avec chiffres",
  "worst_period": "Description de la pire periode avec chiffres",
  "top_products": ["Nom produit 1", "Nom produit 2", "Nom produit 3"],
  "key_segment": "Segment client le plus performant",
  "risks": ["Risque concret 1", "Risque concret 2"],
  "opportunities": ["Opportunite concrete 1", "Opportunite concrete 2"],
  "recommendations": [
    {
      "action": "Action concrete recommandee",
      "priorite": "haute | moyenne | basse",
      "niveau": "strategique | tactique | operationnel",
      "confiance": "elevee | moyenne | faible",
      "justification": "Justification chiffree en une phrase"
    }
  ]
}"""

    schema_en = """{
  "summary": "Global summary sentence in English (1-2 sentences)",
  "main_trend": "Macro trend description",
  "best_period": "Best period description with figures",
  "worst_period": "Worst period description with figures",
  "top_products": ["Product 1", "Product 2", "Product 3"],
  "key_segment": "Top performing customer segment",
  "risks": ["Concrete risk 1", "Concrete risk 2"],
  "opportunities": ["Concrete opportunity 1", "Concrete opportunity 2"],
  "recommendations": [
    {
      "action": "Concrete recommended action",
      "priorite": "haute | moyenne | basse",
      "niveau": "strategique | tactique | operationnel",
      "confiance": "elevee | moyenne | faible",
      "justification": "Quantified justification in one sentence"
    }
  ]
}"""

    if langue == "fr":
        system_msg = (
            "Tu es un analyste business senior. Ton role est d'extraire et structurer "
            "les insights decisionnels d'un dataset de ventes en JSON strict.\n\n"
            "REGLES ABSOLUES :\n"
            "1. Tu reponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni apres.\n"
            "2. Pas de balises ```json``` ni de commentaires.\n"
            "3. Tu n'inventes AUCUN chiffre. Tu utilises uniquement les donnees fournies.\n"
            "4. Les recommandations doivent etre CONCRETES (mentionnent une region, "
            "un produit, un segment ou un trimestre precis).\n"
            "5. Tu produis 3 recommandations minimum, hierarchisees par priorite.\n"
            "6. La 'confiance' depend de la stabilite des donnees (volatilite fournie).\n"
            "7. Le 'niveau' suit cette logique : strategique = touche au modele global, "
            "tactique = touche a un segment/region, operationnel = action court terme.\n"
            "8. Si une donnee est marquee 'non calculable' ou 'N/A', NE l'INVENTE PAS. "
            "Mentionne explicitement que l'information n'est pas disponible.\n\n"
            f"SCHEMA JSON OBLIGATOIRE :\n{schema_fr}"
        )
        user_msg = (
            f"Voici les donnees factuelles a structurer :\n"
            f"{perimetre_block_fr}\n"
            f"Periode : {faits['periode']}\n"
            f"Tendance macro : {tendance}\n"
            f"Croissance globale : {cg_str_fr}\n"
            f"Volatilite trimestrielle : {vol.get('niveau', 'N/A')} "
            f"(moyenne {vol.get('valeur', 0):.1f}%) -> confiance suggeree : {confiance_globale}\n\n"
            f"Evolution annuelle : {annuel_str}\n"
            f"Meilleure annee : {ba['annee']} avec {ba['ca']:,.0f} USD ({ba['commandes']} commandes)\n"
            f"Pire annee : {pa['annee']} avec {pa['ca']:,.0f} USD\n\n"
            f"Meilleur trimestre : T{mt['trimestre']} {mt['annee']} ({mt_var_fr}, {mt['ventes']:,.0f} USD)\n"
            f"Pire trimestre : T{pt['trimestre']} {pt['annee']} ({pt_var_fr}, {pt['ventes']:,.0f} USD)\n\n"
            f"Top 3 sous-categories : "
            f"{t3[0]['nom']} ({t3[0]['ventes']:,.0f} USD), "
            f"{t3[1]['nom']} ({t3[1]['ventes']:,.0f} USD), "
            f"{t3[2]['nom']} ({t3[2]['ventes']:,.0f} USD)\n"
            f"Segment leader : {meilleur_segment[0]} ({meilleur_segment[1]['ventes_total']:,.0f} USD)\n\n"
            f"Genere maintenant le JSON structure (et UNIQUEMENT le JSON)."
        )
    else:
        system_msg = (
            "You are a senior business analyst. Your role is to extract and structure "
            "decision-grade insights from a sales dataset into strict JSON.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Reply with a valid JSON object ONLY, no text before or after.\n"
            "2. No ```json``` fences, no comments.\n"
            "3. NEVER invent figures. Use only provided data.\n"
            "4. Recommendations must be CONCRETE (mention a region, product, "
            "segment or specific quarter).\n"
            "5. Produce at least 3 recommendations, prioritised.\n"
            "6. 'confiance' depends on data stability (volatility given).\n"
            "7. 'niveau' logic: strategique = global model, tactique = segment/region, "
            "operationnel = short-term action.\n"
            "8. If a value is marked 'not computable' or 'N/A', DO NOT INVENT IT. "
            "Explicitly state the information is unavailable.\n\n"
            f"REQUIRED JSON SCHEMA:\n{schema_en}"
        )
        user_msg = (
            f"Factual data to structure:\n"
            f"{perimetre_block_en}\n"
            f"Period: {faits['periode']}\n"
            f"Macro trend: {tendance}\n"
            f"Cumulative growth: {cg_str_en}\n"
            f"Quarterly volatility: {vol.get('niveau', 'N/A')} "
            f"(avg {vol.get('valeur', 0):.1f}%) -> suggested confidence: {confiance_globale}\n\n"
            f"Yearly: {annuel_str}\n"
            f"Best year: {ba['annee']} with ${ba['ca']:,.0f} ({ba['commandes']} orders)\n"
            f"Worst year: {pa['annee']} with ${pa['ca']:,.0f}\n\n"
            f"Best quarter: Q{mt['trimestre']} {mt['annee']} ({mt_var_en}, ${mt['ventes']:,.0f})\n"
            f"Worst quarter: Q{pt['trimestre']} {pt['annee']} ({pt_var_en}, ${pt['ventes']:,.0f})\n\n"
            f"Top 3 sub-categories: "
            f"{t3[0]['nom']} (${t3[0]['ventes']:,.0f}), "
            f"{t3[1]['nom']} (${t3[1]['ventes']:,.0f}), "
            f"{t3[2]['nom']} (${t3[2]['ventes']:,.0f})\n"
            f"Leading segment: {meilleur_segment[0]} (${meilleur_segment[1]['ventes_total']:,.0f})\n\n"
            f"Now generate the structured JSON (and ONLY the JSON)."
        )

    return system_msg, user_msg


def _extraire_json_depuis_reponse(texte: str) -> dict:
    """Extrait un objet JSON d'une reponse LLM (gere fences, texte parasite)."""
    if not texte:
        return None
    try:
        return json.loads(texte.strip())
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", texte)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", texte, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def generer_structure_json(faits: dict, langue: str = "fr") -> dict:
    """Etape 1 sur 2 : genere un JSON structure via Mistral, valide contre les faits."""
    try:
        from config import MISTRAL_API_KEY, MISTRAL_MODEL
    except ImportError:
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
        MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    api_key = MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        logger.warning("MISTRAL_API_KEY absente, structure JSON construite en fallback")
        return _structure_json_fallback(faits, langue)

    try:
        from mistralai.client import Mistral
        client = Mistral(api_key=api_key)
        system_msg, user_msg = _construire_prompt_json(faits, langue)
        logger.info(f"Generation de la structure JSON ({langue.upper()}) via Mistral...")

        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        contenu = response.choices[0].message.content
        structure = _extraire_json_depuis_reponse(contenu)
        if structure is None:
            logger.warning("JSON non parsable, fallback")
            return _structure_json_fallback(faits, langue)

        structure_validee = valider_structure_json(structure, faits, langue)
        logger.info(
            f"[OK] Structure JSON generee et validee "
            f"({len(structure_validee.get('recommendations', []))} recommandations)"
        )
        return structure_validee
    except ImportError:
        logger.warning("mistralai non installe, fallback JSON")
        return _structure_json_fallback(faits, langue)
    except Exception as e:
        logger.warning(f"Erreur Mistral pour JSON: {e} -> fallback")
        return _structure_json_fallback(faits, langue)


# ============================================================
# VALIDATEUR NUMERIQUE (anti-hallucination v2)
# ============================================================
# Le validateur de structure (valider_structure_json) garantit que
# le JSON est bien forme, mais ne controle pas les chiffres dans
# les champs textuels (summary, justifications, prose narrative).
# Le validateur ci-dessous extrait par regex tous les nombres
# trouves dans un texte et les compare a l'ensemble des valeurs
# legitimes derivees des faits Spark. Tout ce qui n'est pas
# attribuable a un fait connu (avec une tolerance d'arrondi) est
# remonte comme suspect.
# ============================================================

_PATTERN_MONTANT = re.compile(
    r"(?<![A-Za-z\d])"
    r"(\$?)"
    r"(\d{1,3}(?:[,\s]\d{3})+|\d+(?:[.,]\d+)?)"
    r"\s*([kKmM])?"
)

_PATTERN_PCT = re.compile(r"([+-]?\d+(?:[.,]\d+)?)\s*%")

_MULT = {"k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000}


def _parser_nombre(brut: str, suffixe: str) -> float:
    """Convertit '234,389' / '1.6' / '62 137' en float, applique le suffixe k/M."""
    nombre = brut.replace(" ", "")
    if "," in nombre and "." in nombre:
        nombre = nombre.replace(",", "")
    elif nombre.count(",") == 1 and len(nombre.split(",")[-1]) <= 2:
        nombre = nombre.replace(",", ".")
    else:
        nombre = nombre.replace(",", "")
    try:
        v = float(nombre)
    except ValueError:
        return float("nan")
    if suffixe:
        v *= _MULT.get(suffixe, 1)
    return v


def _collecter_valeurs_autorisees(faits: dict) -> list:
    """Construit la liste des nombres legitimes derives des faits Spark.
    Retourne [(valeur, label_source)] pour pouvoir tracer l'origine."""
    valeurs = []

    def add(v, label):
        if v is None:
            return
        try:
            valeurs.append((float(v), label))
        except (TypeError, ValueError):
            pass

    # Annuel
    for a in faits.get("annuel", []):
        add(a.get("ca"),             f"CA {a['annee']}")
        add(a.get("commandes"),      f"Commandes {a['annee']}")
        add(a.get("panier_moyen"),   f"Panier moyen {a['annee']}")
        add(a.get("croissance_yoy"), f"YoY {a['annee']}")

    ca_total = sum(a.get("ca") or 0 for a in faits.get("annuel", []))
    add(ca_total, "CA total cumule")
    add(faits.get("croissance_globale"), "Croissance globale")

    # Variations trimestrielles
    # On ajoute aussi la valeur absolue car la prose ecrit souvent
    # "chute de 64,9%" sans le signe negatif.
    for v in faits.get("variations", []):
        add(v.get("ventes"),    f"Ventes T{v['trimestre']} {v['annee']}")
        add(v.get("variation"), f"Variation T{v['trimestre']} {v['annee']}")
        if v.get("variation") is not None:
            add(abs(v["variation"]), f"|Variation| T{v['trimestre']} {v['annee']}")

    # Top produits + combinaisons (Mistral fait souvent Phones+Chairs)
    tops = faits.get("top_produits", []) or []
    for p in tops:
        add(p.get("ventes"), f"Ventes {p['nom']}")
        add(p.get("qte"),    f"Qte {p['nom']}")
    if len(tops) >= 2:
        add(tops[0]["ventes"] + tops[1]["ventes"], f"{tops[0]['nom']}+{tops[1]['nom']}")
    if len(tops) >= 3:
        add(sum(p["ventes"] for p in tops[:3]),
            f"{tops[0]['nom']}+{tops[1]['nom']}+{tops[2]['nom']}")

    # Segments + parts
    segs = faits.get("segments") or {}
    for nom, s in segs.items():
        add(s.get("ventes_total"), f"Ventes segment {nom}")
        add(s.get("commandes"),    f"Commandes segment {nom}")
        if ca_total:
            add(s.get("ventes_total", 0) / ca_total * 100, f"Part segment {nom}")

    # Regions
    regs = faits.get("regions") or {}
    for annee, r in regs.items():
        add(r.get("ventes"), f"Ventes {r.get('region')} {annee}")

    # Mensuel (positions a forte amplitude)
    for m in faits.get("mensuel", []):
        add(m.get("ventes"),    f"Ventes {m['annee']}-{m['mois']:02d}")
        add(m.get("commandes"), f"Commandes {m['annee']}-{m['mois']:02d}")

    # Volatilite
    vol = faits.get("volatilite") or {}
    add(vol.get("valeur"), "Volatilite QoQ moyenne")

    return valeurs


def _proche(valeur: float, ref: float, tolerance_pct: float) -> bool:
    """Compare avec tolerance. Pour |ref| < 5 on tolere une marge absolue de 1.5pt
    (utile sur les YoY autour de zero)."""
    if valeur is None or ref is None:
        return False
    if abs(ref) < 5:
        return abs(valeur - ref) <= 1.5
    return abs(valeur - ref) / abs(ref) * 100 <= tolerance_pct


def valider_chiffres_dans_texte(
    texte: str,
    faits: dict,
    tolerance_pct: float = 4.0,
) -> list:
    """Detecte les nombres du texte non attribuables aux faits Spark.

    Args:
        texte         : prose ou justification a verifier
        faits         : dict de faits Spark
        tolerance_pct : marge d'arrondi acceptee (% de la ref). 4% par defaut.

    Returns:
        list[(extrait_brut, valeur, raison)]
    """
    if not texte:
        return []

    valeurs_ref = _collecter_valeurs_autorisees(faits)
    suspects = []

    # ---- 1. Montants (avec indice monetaire $, k, M ou USD a proximite) ----
    for m in _PATTERN_MONTANT.finditer(texte):
        prefixe_dollar, nombre, suffixe = m.group(1), m.group(2), m.group(3)
        contexte = texte[max(0, m.start()-2):m.end()+6]
        a_indice = bool(prefixe_dollar or suffixe) or \
                   "USD" in contexte or "$" in contexte or "EUR" in contexte
        if not a_indice:
            continue
        valeur = _parser_nombre(nombre, suffixe or "")
        if valeur != valeur:  # NaN
            continue
        # Annees pures (2015-2030) sans suffixe ni $ -> on ignore
        if 2010 <= valeur <= 2030 and not suffixe and not prefixe_dollar \
                and "USD" not in contexte and "$" not in contexte:
            continue
        if any(_proche(valeur, ref, tolerance_pct) for ref, _ in valeurs_ref):
            continue
        suspects.append((m.group(0).strip(), valeur, "montant non trouve"))

    # ---- 2. Pourcentages : tolerance plus stricte (1.5%) car ces
    #         chiffres sont souvent issus de KPIs precis, peu d'arrondis.
    tolerance_pct_pourcent = min(tolerance_pct, 1.5)
    for m in _PATTERN_PCT.finditer(texte):
        try:
            valeur = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        if any(_proche(valeur, ref, tolerance_pct_pourcent) for ref, _ in valeurs_ref):
            continue
        suspects.append((m.group(0).strip(), valeur, "pourcentage non trouve"))

    return suspects


def _generer_justification_safe(reco: dict, faits: dict, langue: str) -> str:
    """Justification deterministe basee sur les faits, utilisee quand la
    justification originale contient des chiffres non valides."""
    vol = faits.get("volatilite") or {}
    tops = faits.get("top_produits") or []
    pt = faits.get("pire_trim") or {}
    mt = faits.get("meilleur_trim") or {}
    if langue == "fr":
        if "volatilit" in reco.get("action", "").lower():
            return (f"Volatilite trimestrielle moyenne de {vol.get('valeur', 0):.1f}% "
                    f"(niveau {vol.get('niveau', 'modere').lower()}).")
        if tops:
            return (f"La sous-categorie {tops[0]['nom']} represente "
                    f"{tops[0]['ventes']:,.0f} USD sur la periode analysee.")
        return f"Base sur les faits Spark de la periode {faits.get('periode', '')}."
    if "volatil" in reco.get("action", "").lower():
        return (f"Average quarterly volatility of {vol.get('valeur', 0):.1f}% "
                f"({vol.get('niveau', 'moderate').lower()} level).")
    if tops:
        return (f"The {tops[0]['nom']} sub-category represents "
                f"${tops[0]['ventes']:,.0f} over the analysed period.")
    return f"Based on Spark facts for the {faits.get('periode', '')} period."


def valider_structure_json(structure: dict, faits: dict, langue: str = "fr") -> dict:
    """Verifie que la structure est complete. Complete avec defauts si necessaire."""
    if not isinstance(structure, dict):
        return _structure_json_fallback(faits, langue)

    champs_str = ["summary", "main_trend", "best_period", "worst_period", "key_segment"]
    for champ in champs_str:
        if not isinstance(structure.get(champ), str) or not structure.get(champ).strip():
            structure[champ] = _valeur_defaut_champ(champ, faits, langue)

    if not isinstance(structure.get("top_products"), list) or len(structure["top_products"]) < 3:
        structure["top_products"] = [p["nom"] for p in faits["top_produits"][:3]]

    if not isinstance(structure.get("risks"), list) or len(structure["risks"]) == 0:
        structure["risks"] = _risques_defaut(faits, langue)

    if not isinstance(structure.get("opportunities"), list) or len(structure["opportunities"]) == 0:
        structure["opportunities"] = _opportunites_defaut(faits, langue)

    if not isinstance(structure.get("recommendations"), list):
        structure["recommendations"] = []

    valeurs_priorite = {"haute", "moyenne", "basse"}
    valeurs_niveau = {"strategique", "tactique", "operationnel"}
    valeurs_confiance = {"elevee", "moyenne", "faible"}

    recos_validees = []
    for r in structure["recommendations"]:
        if not isinstance(r, dict) or not r.get("action"):
            continue
        r["priorite"] = r.get("priorite", "moyenne").lower()
        if r["priorite"] not in valeurs_priorite:
            r["priorite"] = "moyenne"
        r["niveau"] = r.get("niveau", "tactique").lower()
        if r["niveau"] not in valeurs_niveau:
            r["niveau"] = "tactique"
        r["confiance"] = r.get("confiance", "moyenne").lower()
        if r["confiance"] not in valeurs_confiance:
            r["confiance"] = "moyenne"
        if not r.get("justification"):
            r["justification"] = "Base sur les donnees analysees." if langue == "fr" else "Based on analysed data."
        else:
            # Validation numerique de la justification.
            suspects = valider_chiffres_dans_texte(r["justification"], faits)
            if suspects:
                logger.warning(
                    f"Justification reco suspecte ({len(suspects)} chiffre(s)) -> "
                    f"reecriture deterministe. Suspects: {suspects[:3]}"
                )
                r["justification"] = _generer_justification_safe(r, faits, langue)

        # Validation numerique de l'action elle-meme : Mistral glisse
        # parfois des chiffres derives (ex "45.7% du total") qu'il
        # faut nettoyer.
        if r.get("action"):
            action_suspects = valider_chiffres_dans_texte(r["action"], faits)
            if action_suspects:
                logger.warning(
                    f"Action reco suspecte ({len(action_suspects)} chiffre(s)) -> "
                    f"nettoyage. Suspects: {action_suspects[:3]}"
                )
                # On retire chaque extrait suspect du texte d'action.
                action_clean = r["action"]
                for extrait, _, _ in action_suspects:
                    # Retire "(45.7% du total)" ou "(45.7%)" qui entoure souvent ces chiffres
                    for pattern_paren in [f"({extrait} du total)", f"({extrait})", extrait]:
                        if pattern_paren in action_clean:
                            action_clean = action_clean.replace(pattern_paren, "").strip()
                            break
                # Nettoyer les doubles espaces et virgules orphelines
                action_clean = re.sub(r"\s+", " ", action_clean)
                action_clean = re.sub(r"\s+,", ",", action_clean)
                action_clean = re.sub(r",\s*\.", ".", action_clean)
                r["action"] = action_clean.strip()

        recos_validees.append(r)

    if len(recos_validees) < 2:
        recos_validees.extend(_recommandations_defaut(faits, langue))

    ordre_prio = {"haute": 0, "moyenne": 1, "basse": 2}
    recos_validees.sort(key=lambda r: ordre_prio.get(r["priorite"], 99))
    structure["recommendations"] = recos_validees[:5]
    return structure


def _valeur_defaut_champ(champ: str, faits: dict, langue: str) -> str:
    """Valeur defaut pour un champ texte manquant."""
    ba = faits["meilleure_annee"]
    pt = faits["pire_trim"]
    cg_raw = faits.get("croissance_globale")
    tendance = faits.get("tendance_globale", {}).get("label", "stable")

    cg_str_fr = f"{cg_raw:+.1f}%" if cg_raw is not None else "non calculable"
    cg_str_en = f"{cg_raw:+.1f}%" if cg_raw is not None else "not computable"
    pt_var = pt.get("variation")
    pt_var_fr = f"{pt_var:+.1f}%" if pt_var is not None else "QoQ N/A"
    pt_var_en = f"{pt_var:+.1f}%" if pt_var is not None else "QoQ N/A"

    meilleur_segment = max(
        faits["segments"].items(),
        key=lambda x: x[1]["ventes_total"],
    )[0] if faits.get("segments") else "N/A"

    if langue == "fr":
        if cg_raw is None:
            summary_txt = (
                f"Sur la periode {faits['periode']}, l'activite affiche un chiffre "
                f"d'affaires de {ba['ca']:,.0f} USD (croissance YoY non calculable)."
            )
        else:
            summary_txt = (
                f"Sur la periode {faits['periode']}, l'activite affiche une croissance "
                f"de {cg_str_fr} avec une tendance {tendance.lower()}."
            )
        defauts = {
            "summary": summary_txt,
            "main_trend": tendance.lower(),
            "best_period": f"{ba['annee']} avec {ba['ca']:,.0f} USD",
            "worst_period": f"T{pt['trimestre']} {pt['annee']} ({pt_var_fr})",
            "key_segment": meilleur_segment,
        }
    else:
        if cg_raw is None:
            summary_txt = (
                f"Over {faits['periode']}, activity shows ${ba['ca']:,.0f} in "
                f"revenue (YoY growth not computable)."
            )
        else:
            summary_txt = (
                f"Over {faits['periode']}, activity shows {cg_str_en} growth "
                f"with a {tendance.lower()} trend."
            )
        defauts = {
            "summary": summary_txt,
            "main_trend": tendance.lower(),
            "best_period": f"{ba['annee']} with ${ba['ca']:,.0f}",
            "worst_period": f"Q{pt['trimestre']} {pt['annee']} ({pt_var_en})",
            "key_segment": meilleur_segment,
        }
    return defauts.get(champ, "")


def _risques_defaut(faits: dict, langue: str) -> list:
    """Risques par defaut robustes aux variations None."""
    t1 = faits["top_produits"][0]
    vol = faits.get("volatilite", {})
    pt = faits["pire_trim"]
    pt_var = pt.get("variation")
    pt_var_fr = f"{pt_var:+.1f}%" if pt_var is not None else "QoQ N/A"
    pt_var_en = f"{pt_var:+.1f}%" if pt_var is not None else "QoQ N/A"

    if langue == "fr":
        return [
            f"Dependance forte a la sous-categorie {t1['nom']} ({t1['ventes']:,.0f} USD).",
            f"Volatilite trimestrielle {vol.get('niveau', 'moderee').lower()} "
            f"(variation moyenne {vol.get('valeur', 0):.1f}%).",
            f"Trimestre fragile recurrent : T{pt['trimestre']} ({pt_var_fr}).",
        ]
    return [
        f"Strong dependence on sub-category {t1['nom']} (${t1['ventes']:,.0f}).",
        f"Quarterly volatility {vol.get('niveau', 'moderate').lower()} "
        f"(avg variation {vol.get('valeur', 0):.1f}%).",
        f"Recurring weak quarter: Q{pt['trimestre']} ({pt_var_en}).",
    ]


def _opportunites_defaut(faits: dict, langue: str) -> list:
    """Opportunites par defaut robustes aux variations None."""
    mt = faits["meilleur_trim"]
    mt_var = mt.get("variation")
    mt_var_fr = f"{mt_var:+.1f}%" if mt_var is not None else "QoQ N/A"
    mt_var_en = mt_var_fr

    meilleur_segment = max(
        faits["segments"].items(),
        key=lambda x: x[1]["ventes_total"],
    ) if faits.get("segments") else ("N/A", {"ventes_total": 0})

    if langue == "fr":
        return [
            f"Capitaliser sur le pic du T{mt['trimestre']} ({mt_var_fr}).",
            f"Renforcer le segment {meilleur_segment[0]} "
            f"({meilleur_segment[1]['ventes_total']:,.0f} USD).",
        ]
    return [
        f"Capitalize on Q{mt['trimestre']} peak ({mt_var_en}).",
        f"Reinforce segment {meilleur_segment[0]} "
        f"(${meilleur_segment[1]['ventes_total']:,.0f}).",
    ]


def _recommandations_defaut(faits: dict, langue: str) -> list:
    """Recommandations defaut robustes aux variations None."""
    t1 = faits["top_produits"][0]
    mt = faits["meilleur_trim"]
    pt = faits["pire_trim"]
    vol = faits.get("volatilite", {})

    mt_var = mt.get("variation")
    pt_var = pt.get("variation")
    mt_var_fr = f"{mt_var:+.1f}%" if mt_var is not None else "QoQ N/A"
    pt_var_fr = f"{pt_var:+.1f}%" if pt_var is not None else "QoQ N/A"
    mt_var_en = mt_var_fr
    pt_var_en = pt_var_fr

    vol_str = vol.get("niveau", "Moderee").lower()
    if "faible" in vol_str:
        conf = "elevee"
    elif "elevee" in vol_str or "tres" in vol_str:
        conf = "faible"
    else:
        conf = "moyenne"

    if langue == "fr":
        return [
            {
                "action": f"Renforcer les stocks et le marketing sur le T{mt['trimestre']} "
                          f"qui concentre les meilleures performances historiques.",
                "priorite": "haute",
                "niveau": "tactique",
                "confiance": conf,
                "justification": f"T{mt['trimestre']} affiche {mt_var_fr} de variation, signal recurrent.",
            },
            {
                "action": f"Lancer un plan de relance dedie au T{pt['trimestre']} pour lisser la saisonnalite.",
                "priorite": "haute",
                "niveau": "tactique",
                "confiance": conf,
                "justification": f"T{pt['trimestre']} reste le creux recurrent ({pt_var_fr}).",
            },
            {
                "action": f"Diversifier l'offre au-dela de {t1['nom']} pour reduire la dependance produit.",
                "priorite": "moyenne",
                "niveau": "strategique",
                "confiance": "moyenne",
                "justification": f"{t1['nom']} concentre une part dominante du CA.",
            },
        ]
    return [
        {
            "action": f"Reinforce inventory and marketing on Q{mt['trimestre']} "
                      f"which historically delivers the best performance.",
            "priorite": "haute",
            "niveau": "tactique",
            "confiance": conf,
            "justification": f"Q{mt['trimestre']} shows {mt_var_en} variation, a recurring signal.",
        },
        {
            "action": f"Launch a dedicated recovery plan for Q{pt['trimestre']} to smooth seasonality.",
            "priorite": "haute",
            "niveau": "tactique",
            "confiance": conf,
            "justification": f"Q{pt['trimestre']} remains the recurring trough ({pt_var_en}).",
        },
        {
            "action": f"Diversify offering beyond {t1['nom']} to reduce product dependence.",
            "priorite": "moyenne",
            "niveau": "strategique",
            "confiance": "moyenne",
            "justification": f"{t1['nom']} concentrates a dominant share of revenue.",
        },
    ]


def _structure_json_fallback(faits: dict, langue: str) -> dict:
    """Structure JSON complete sans appel Mistral."""
    return {
        "summary": _valeur_defaut_champ("summary", faits, langue),
        "main_trend": _valeur_defaut_champ("main_trend", faits, langue),
        "best_period": _valeur_defaut_champ("best_period", faits, langue),
        "worst_period": _valeur_defaut_champ("worst_period", faits, langue),
        "top_products": [p["nom"] for p in faits["top_produits"][:3]],
        "key_segment": _valeur_defaut_champ("key_segment", faits, langue),
        "risks": _risques_defaut(faits, langue),
        "opportunities": _opportunites_defaut(faits, langue),
        "recommendations": _recommandations_defaut(faits, langue),
    }


def _generer_texte_depuis_structure(structure: dict, faits: dict, langue: str) -> str:
    """Etape 2 sur 2 : transforme le JSON valide en 4 paragraphes de prose."""
    try:
        from config import MISTRAL_API_KEY, MISTRAL_MODEL
    except ImportError:
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
        MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    api_key = MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        return _texte_depuis_structure_fallback(structure, faits, langue)

    structure_json_str = json.dumps(structure, ensure_ascii=False, indent=2)

    if langue == "fr":
        system_msg = (
            "Tu es directeur business intelligence chez Superstore. Tu recois une "
            "structure JSON validee contenant les insights cles et tu dois la "
            "transformer en UN TEXTE NARRATIF CONTINU en francais, comme un briefing "
            "a un dirigeant.\n\n"
            "INTERDICTIONS ABSOLUES :\n"
            "- Aucun caractere * # _\n"
            "- Aucun titre de section, aucune liste a puces, aucune numerotation\n"
            "- Aucun mot en MAJUSCULES (sauf USD, CA, T1-T4, YoY, TCAM)\n"
            "- Aucun prefixe 'Analyse :' ou 'Recommandations :'\n"
            "- NE JAMAIS inventer un chiffre. Si une donnee n'est pas dans la structure, "
            "ne la cite pas.\n\n"
            "STYLE DE REASONING (essentiel) :\n"
            "Tu ne dois pas te limiter a des constats descriptifs. Pour chaque fait "
            "important, tu dois enchainer : constat -> implication metier -> "
            "risque ou opportunite associee -> piste d'action.\n"
            "Exemple a ne PAS faire : 'La region West domine.'\n"
            "Exemple a faire : 'La region West concentre la majorite du CA, ce qui "
            "indique une dependance geographique. Cette concentration cree un risque "
            "si West ralentit ; East et Central representent des relais de croissance "
            "encore sous-exploites.'\n\n"
            "REGLES METIER A APPLIQUER :\n"
            "- Une concentration > 35% sur un axe = signaler le risque de dependance.\n"
            "- Une croissance forte (>30%) = se demander si elle est soutenable.\n"
            "- Une baisse trimestrielle = chercher la cause potentielle (saisonnalite, marche, ops).\n"
            "- Un panier moyen qui baisse alors que les commandes augmentent = signal d'upsell rate.\n"
            "- Une volatilite > 30% = trop de bruit pour piloter sereinement.\n\n"
            "FORMAT OBLIGATOIRE :\n"
            "- 4 paragraphes de prose fluide separes par une ligne vide\n"
            "- Chaque paragraphe = 4 a 6 phrases completes\n"
            "- Paragraphe 1 : contexte global + tendance + lecture macro (capitalisation/prudence)\n"
            "- Paragraphe 2 : meilleures et pires periodes + interpretation (cycles, saisonnalite, structurel)\n"
            "- Paragraphe 3 : produits + segment leader + risques de dependance + diversification\n"
            "- Paragraphe 4 : recommandations en prose, chaque action liee a son impact attendu\n\n"
            "Tu integres tous les chiffres presents dans la structure naturellement. "
            "Tu connectes les paragraphes logiquement : chaque suivant prolonge le precedent."
        )
        user_msg = (
            f"Voici la structure JSON validee a transformer en prose narrative :\n\n"
            f"{structure_json_str}\n\n"
            f"Redige maintenant les 4 paragraphes. Commence directement "
            f"par 'Sur la periode {faits['periode']}' ou equivalent."
        )
    else:
        system_msg = (
            "You are a business intelligence director at Superstore. You receive a "
            "validated JSON structure with key insights and must turn it into ONE "
            "CONTINUOUS NARRATIVE TEXT in English, like an executive briefing.\n\n"
            "ABSOLUTE BANS:\n"
            "- No * # _ characters\n"
            "- No section headings, no bullet lists, no numbering\n"
            "- No UPPERCASE words (except USD, CAGR, Q1-Q4, YoY)\n"
            "- No 'Analysis:' or 'Recommendations:' prefixes\n"
            "- NEVER invent a number. If data is not in the structure, do not cite it.\n\n"
            "REASONING STYLE (critical):\n"
            "Don't stop at descriptive statements. For each important fact, chain: "
            "observation -> business implication -> related risk or opportunity -> action lead.\n"
            "Bad example: 'West region dominates.'\n"
            "Good example: 'The West region concentrates the majority of revenue, "
            "indicating geographic dependence. This concentration creates a risk if "
            "West slows down; East and Central represent under-exploited growth relays.'\n\n"
            "BUSINESS RULES TO APPLY:\n"
            "- Concentration > 35% on any axis = flag dependence risk.\n"
            "- Strong growth (>30%) = question sustainability.\n"
            "- Quarterly decline = explore root cause (seasonality, market, ops).\n"
            "- Average basket dropping while orders rising = missed upsell signal.\n"
            "- Volatility > 30% = too much noise for stable steering.\n\n"
            "REQUIRED FORMAT:\n"
            "- 4 flowing paragraphs separated by a blank line\n"
            "- Each paragraph = 4 to 6 full sentences\n"
            "- Paragraph 1: global context + trend + macro reading\n"
            "- Paragraph 2: best and worst periods + interpretation (cycles, seasonality, structural)\n"
            "- Paragraph 3: products + leading segment + dependence risks + diversification\n"
            "- Paragraph 4: recommendations as prose, each action linked to its expected impact\n\n"
            "Integrate all figures from the structure naturally. Chain paragraphs logically."
        )
        user_msg = (
            f"Here is the validated JSON structure to turn into narrative prose:\n\n"
            f"{structure_json_str}\n\n"
            f"Now write the 4 paragraphs. Start directly with 'Over the period "
            f"{faits['periode']}' or equivalent."
        )

    try:
        from mistralai.client import Mistral
        client = Mistral(api_key=api_key)
        logger.info(f"Generation du texte narratif depuis JSON ({langue.upper()})...")
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=1100,
        )
        rapport = response.choices[0].message.content
        rapport = _strip_markdown(rapport)
        if len(rapport) < 400 or rapport.strip().startswith(("RAPPORT", "CONTEXTE", "PERFORMANCES")):
            logger.warning(f"Rapport narratif trop court ({len(rapport)} chars), fallback")
            return _texte_depuis_structure_fallback(structure, faits, langue)
        return rapport
    except Exception as e:
        logger.warning(f"Erreur generation narrative: {e} -> fallback")
        return _texte_depuis_structure_fallback(structure, faits, langue)


def _texte_depuis_structure_fallback(structure: dict, faits: dict, langue: str) -> str:
    """Texte narratif sans Mistral, garanti coherent avec la structure validee."""
    s = structure
    recos = s.get("recommendations", [])

    if langue == "fr":
        p1 = (
            f"{s['summary']} La tendance de fond reste {s['main_trend']}, "
            f"ce qui definit le cadre de lecture des resultats. "
            f"Cette dynamique combine des moments forts et des points de vigilance "
            f"qu'il convient d'analyser conjointement pour eclairer le pilotage. "
            f"Le contexte global appelle donc a la fois capitalisation et prudence."
        )
        p2 = (
            f"La meilleure periode identifiee est {s['best_period']}, qui marque "
            f"un point d'ancrage important pour la trajectoire. "
            f"A l'inverse, {s['worst_period']} constitue le creux le plus marque "
            f"et revele une fragilite recurrente. "
            f"L'ecart entre ces deux extremes illustre l'importance d'une strategie "
            f"de lissage et d'anticipation des cycles."
        )
        risques_str = " ".join(s.get("risks", [])[:2])
        p3 = (
            f"Les sous-categories les plus contributrices sont "
            f"{', '.join(s['top_products'][:3])}, ce qui dessine clairement la "
            f"colonne vertebrale commerciale. "
            f"Le segment client le plus performant reste {s['key_segment']}, "
            f"confirmant son role moteur. "
            f"{risques_str} "
            f"Ces points appellent une vigilance active dans les mois a venir."
        )
        if recos:
            phrases_reco = []
            connecteurs = ["En priorite", "Par ailleurs", "Enfin"]
            for i, r in enumerate(recos[:3]):
                conn = connecteurs[i] if i < len(connecteurs) else "Egalement"
                phrases_reco.append(
                    f"{conn}, {r['action'].lower()[0] + r['action'][1:] if r['action'] else 'agir.'}"
                )
            p4 = " ".join(phrases_reco) + (
                " Ces actions, combinees, devraient renforcer la resilience globale "
                "tout en exploitant les leviers identifies dans l'analyse."
            )
        else:
            p4 = (
                "Pour consolider ces resultats, il convient d'agir simultanement "
                "sur les pics saisonniers, les creux recurrents et la diversification "
                "produits. Une approche coordonnee renforcera la resilience globale."
            )
        return f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}"

    p1 = (
        f"{s['summary']} The underlying trend remains {s['main_trend']}, "
        f"which sets the reading frame for the results. "
        f"This dynamic combines strong moments and watch-points that need "
        f"to be analysed jointly to inform decision-making. "
        f"The overall context calls for both capitalisation and caution."
    )
    p2 = (
        f"The best period identified is {s['best_period']}, marking a key "
        f"anchor point for the trajectory. "
        f"Conversely, {s['worst_period']} represents the most pronounced trough "
        f"and reveals a recurring weakness. "
        f"The gap between these extremes shows the importance of smoothing "
        f"and cycle anticipation."
    )
    risks_str = " ".join(s.get("risks", [])[:2])
    p3 = (
        f"The top contributing sub-categories are {', '.join(s['top_products'][:3])}, "
        f"clearly forming the commercial backbone. "
        f"The leading customer segment remains {s['key_segment']}, confirming "
        f"its driving role. "
        f"{risks_str} "
        f"These points call for active monitoring in the months ahead."
    )
    if recos:
        phrases_reco = []
        connecteurs = ["As a priority", "In addition", "Finally"]
        for i, r in enumerate(recos[:3]):
            conn = connecteurs[i] if i < len(connecteurs) else "Also"
            phrases_reco.append(
                f"{conn}, {r['action'].lower()[0] + r['action'][1:] if r['action'] else 'act.'}"
            )
        p4 = " ".join(phrases_reco) + (
            " These combined actions should strengthen overall resilience "
            "while exploiting the levers identified in the analysis."
        )
    else:
        p4 = (
            "To build on these results, act simultaneously on seasonal peaks, "
            "recurring troughs and product diversification. A coordinated "
            "approach will strengthen overall resilience."
        )
    return f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}"


def generer_rapport(faits: dict, langue: str = "fr") -> str:
    """Pipeline 2 etapes : structure JSON puis prose narrative."""
    logger.info(f"Generation du rapport en 2 etapes ({langue.upper()})...")
    structure = generer_structure_json(faits, langue=langue)
    rapport = _generer_texte_depuis_structure(structure, faits, langue)
    rapport = _strip_markdown(rapport)
    if len(rapport) < 400:
        logger.warning(f"Rapport final trop court ({len(rapport)} chars), fallback total")
        rapport = _strip_markdown(_generer_rapport_fallback(faits, langue))
    # Validation numerique de la prose : si trop d'hallucinations, on tombe
    # sur le fallback deterministe qui derive sa prose de la structure validee.
    suspects = valider_chiffres_dans_texte(rapport, faits)
    if suspects:
        logger.warning(
            f"Prose Mistral : {len(suspects)} chiffre(s) suspect(s) detecte(s). "
            f"Exemples: {[(s[0], s[2]) for s in suspects[:5]]}"
        )
        if len(suspects) >= 3:
            logger.warning("Seuil 3+ chiffres suspects -> bascule sur fallback prose.")
            rapport = _strip_markdown(_texte_depuis_structure_fallback(structure, faits, langue))
    logger.info(f"[OK] Rapport final genere ({len(rapport)} chars)")
    return rapport


def generer_rapport_avec_structure(faits: dict, langue: str = "fr") -> dict:
    """Variante retournant rapport + structure JSON ensemble."""
    structure = generer_structure_json(faits, langue=langue)
    rapport = _generer_texte_depuis_structure(structure, faits, langue)
    rapport = _strip_markdown(rapport)
    if len(rapport) < 400:
        rapport = _strip_markdown(_generer_rapport_fallback(faits, langue))
    suspects = valider_chiffres_dans_texte(rapport, faits)
    if suspects and len(suspects) >= 3:
        logger.warning(
            f"Prose Mistral : {len(suspects)} chiffres suspects -> fallback prose."
        )
        rapport = _strip_markdown(_texte_depuis_structure_fallback(structure, faits, langue))
    return {"rapport": rapport, "structure": structure, "chiffres_suspects": suspects}


def _generer_rapport_fallback(faits: dict, langue: str) -> str:
    """Rapport secours sans Mistral, prose longue en 4 paragraphes."""
    ba = faits["meilleure_annee"]
    mt = faits["meilleur_trim"]
    pt = faits["pire_trim"]
    t3 = faits.get("top_produits", [])
    vol = faits.get("volatilite", {})
    tendance = faits.get("tendance_globale", {}).get("label", "stable")
    meilleur_segment = max(
        faits.get("segments", {}).items(),
        key=lambda x: x[1]["ventes_total"],
    ) if faits.get("segments") else ("N/A", {"ventes_total": 0})

    cg = faits.get("croissance_globale")
    cg_str = f"{cg:+.1f}%" if cg is not None else "non calculable (pas d'annee comparable)"

    mt_var = mt.get("variation")
    pt_var = pt.get("variation")
    mt_var_str_fr = f"{mt_var:+.1f}%" if mt_var is not None else "variation QoQ non calculable"
    pt_var_str_fr = f"{pt_var:+.1f}%" if pt_var is not None else "variation QoQ non calculable"
    mt_var_str_en = f"{mt_var:+.1f}%" if mt_var is not None else "QoQ variation N/A"
    pt_var_str_en = f"{pt_var:+.1f}%" if pt_var is not None else "QoQ variation N/A"

    top_noms_fr = ", ".join(
        [f"{p['nom']} ({p['ventes']:,.0f} USD)" for p in t3[:3]]
    ) if t3 else "non disponibles"

    if langue == "fr":
        p1 = (
            f"Sur la periode {faits['periode']}, l'activite Superstore affiche une tendance "
            f"{tendance.lower()} avec une croissance cumulee de {cg_str}. "
            f"La volatilite trimestrielle reste {vol.get('niveau', 'moderee').lower()} "
            f"(variation moyenne de {vol.get('valeur', 0):.1f}%), ce qui traduit une activite "
            f"soumise a des cycles saisonniers marques. Cette instabilite cree a la fois des "
            f"opportunites de capitalisation sur les pics et des points de vigilance sur les "
            f"creux. Le contexte global reste donc porteur mais exige un pilotage fin des "
            f"trimestres de transition."
        )
        p2 = (
            f"La meilleure annee reste {ba['annee']} avec {ba['ca']:,.0f} USD de chiffre d'affaires "
            f"repartis sur {ba['commandes']} commandes, confirmant la robustesse du modele. "
            f"Le trimestre le plus dynamique a ete T{mt['trimestre']} {mt['annee']} avec une "
            f"variation de {mt_var_str_fr} ({mt['ventes']:,.0f} USD). A l'inverse, "
            f"T{pt['trimestre']} {pt['annee']} a marque un creux avec {pt_var_str_fr} "
            f"({pt['ventes']:,.0f} USD), soulignant une fragilite recurrente."
        )
        p3 = (
            f"Les sous-categories les plus contributrices sont {top_noms_fr}, ce qui traduit "
            f"une dependance forte a un petit nombre de lignes produits. Le segment le plus "
            f"performant est {meilleur_segment[0]} avec "
            f"{meilleur_segment[1]['ventes_total']:,.0f} USD. "
            f"Cette concentration represente un atout commercial mais aussi un risque de "
            f"dependance qu'il convient de surveiller."
        )
        p4 = (
            f"Pour consolider ces resultats, il serait pertinent de capitaliser sur les pics "
            f"saisonniers identifies, notamment en renforcant les stocks et les campagnes "
            f"marketing sur les trimestres historiquement porteurs. Un plan dedie aux premiers "
            f"trimestres, souvent atones, permettrait de lisser la courbe d'activite et "
            f"d'ameliorer la previsibilite. Enfin, un elargissement progressif vers des "
            f"sous-categories complementaires renforcerait la resilience globale."
        )
        return f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}"

    top_noms_en = ", ".join(
        [f"{p['nom']} (${p['ventes']:,.0f})" for p in t3[:3]]
    ) if t3 else "not available"

    p1 = (
        f"Over the period {faits['periode']}, Superstore's activity shows a {tendance.lower()} "
        f"trend with cumulative growth of {cg_str}. Quarterly volatility "
        f"remains {vol.get('niveau', 'moderate').lower()} "
        f"(avg variation {vol.get('valeur', 0):.1f}%)."
    )
    p2 = (
        f"The best year remains {ba['annee']} with ${ba['ca']:,.0f} in revenue across "
        f"{ba['commandes']} orders. The strongest quarter was Q{mt['trimestre']} {mt['annee']} "
        f"at {mt_var_str_en} (${mt['ventes']:,.0f}). Conversely, Q{pt['trimestre']} {pt['annee']} "
        f"hit a low at {pt_var_str_en} (${pt['ventes']:,.0f})."
    )
    p3 = (
        f"Top sub-categories are {top_noms_en}, showing strong dependence on a narrow set of "
        f"product lines. The leading segment is {meilleur_segment[0]} with "
        f"${meilleur_segment[1]['ventes_total']:,.0f}."
    )
    p4 = (
        f"To build on these results, capitalize on identified seasonal peaks by reinforcing "
        f"stocks and marketing campaigns on historically strong quarters. A dedicated plan "
        f"for the typically flat first quarters would smooth the activity curve and improve "
        f"predictability."
    )
    return f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}"


def generer_resume_executif(faits: dict, langue: str = "fr") -> dict:
    """Resume executif en 3 bullets robustes aux valeurs None."""
    ba = faits["meilleure_annee"]
    mt = faits["meilleur_trim"]
    t1 = faits["top_produits"][0] if faits.get("top_produits") else {"nom": "N/A", "ventes": 0, "qte": 0}
    cg_raw = faits.get("croissance_globale")
    periode = faits["periode"]

    anomalie = _detecter_anomalie_principale(faits)

    if cg_raw is None:
        bullet1_fr = (
            f"Chiffre d'affaires de {ba['ca']:,.0f} USD en {ba['annee']} "
            f"sur la periode {periode}. Croissance YoY non calculable "
            f"(pas d'annee comparable de reference)."
        )
        bullet1_en = (
            f"Revenue of ${ba['ca']:,.0f} in {ba['annee']} over period {periode}. "
            f"YoY growth not computable (no comparable reference year)."
        )
    else:
        cg = float(cg_raw)
        bullet1_fr = (
            f"Croissance globale de {cg:+.1f}% sur la periode {periode}, "
            f"avec un chiffre d'affaires record de {ba['ca']:,.0f} USD en {ba['annee']}."
        )
        bullet1_en = (
            f"Overall revenue grew {cg:+.1f}% over the period {periode}, "
            f"reaching a record ${ba['ca']:,.0f} in {ba['annee']}."
        )

    mt_var = mt.get("variation")
    if mt_var is None:
        bullet2_fr = (
            f"Trimestre le plus actif : T{mt['trimestre']} {mt['annee']} "
            f"avec {mt['ventes']:,.0f} USD de ventes, porte par la categorie {t1['nom']} "
            f"({t1['ventes']:,.0f} USD)."
        )
        bullet2_en = (
            f"Most active quarter: Q{mt['trimestre']} {mt['annee']} "
            f"with ${mt['ventes']:,.0f} in sales, driven by {t1['nom']} "
            f"(${t1['ventes']:,.0f})."
        )
    else:
        bullet2_fr = (
            f"Meilleure performance trimestrielle : T{mt['trimestre']} {mt['annee']} "
            f"({mt_var:+.1f}%), portee par la categorie {t1['nom']} "
            f"({t1['ventes']:,.0f} USD)."
        )
        bullet2_en = (
            f"Best quarter: Q{mt['trimestre']} {mt['annee']} "
            f"({mt_var:+.1f}%), driven by {t1['nom']} "
            f"(${t1['ventes']:,.0f})."
        )

    if langue == "fr":
        bullets = [bullet1_fr, bullet2_fr, anomalie["fr"]]
        titre = "RESUME EXECUTIF"
    else:
        bullets = [bullet1_en, bullet2_en, anomalie["en"]]
        titre = "EXECUTIVE SUMMARY"

    return {"titre": titre, "bullets": bullets}


def detecter_anomalies(faits: dict, seuil_alerte: float = None) -> list:
    """
    Detection YoY. Compare meme trimestre annee precedente.
    Seuil par defaut +/- 20% (canonique du projet,
    aligne config.ANOMALY_THRESHOLD et db.get_anomalies).
    """
    seuil = seuil_alerte if seuil_alerte is not None else ANOMALY_THRESHOLD
    anomalies = []

    variations_by_quarter = {}
    for v in faits["variations"]:
        trim = v["trimestre"]
        annee = v["annee"]
        if trim not in variations_by_quarter:
            variations_by_quarter[trim] = {}
        variations_by_quarter[trim][annee] = v

    for trim in sorted(variations_by_quarter.keys()):
        annees = sorted(variations_by_quarter[trim].keys())
        for i in range(1, len(annees)):
            annee_actuelle = annees[i]
            annee_precedente = annees[i - 1]
            if annee_actuelle - annee_precedente != 1:
                continue
            v_actuelle = variations_by_quarter[trim][annee_actuelle]
            v_precedente = variations_by_quarter[trim][annee_precedente]
            ventes_actuelle = v_actuelle["ventes"]
            ventes_precedente = v_precedente["ventes"]
            if ventes_precedente > 0:
                variation_yoy = ((ventes_actuelle - ventes_precedente) / ventes_precedente) * 100
            else:
                variation_yoy = 0.0

            if variation_yoy <= -seuil:
                anomalies.append({
                    "niveau": "ALERTE",
                    "type": "chute_yoy",
                    "annee": annee_actuelle, "trimestre": trim,
                    "variation": variation_yoy,
                    "ventes": ventes_actuelle, "ventes_ref": ventes_precedente,
                    "annee_ref": annee_precedente,
                    "fr": (f"ALERTE - Chute de {abs(variation_yoy):.1f}% au "
                           f"T{trim} {annee_actuelle} ({ventes_actuelle:,.0f} USD) vs "
                           f"T{trim} {annee_precedente} ({ventes_precedente:,.0f} USD). "
                           f"Analyse recommandee."),
                    "en": (f"ALERT - {abs(variation_yoy):.1f}% YoY drop in "
                           f"Q{trim} {annee_actuelle} (${ventes_actuelle:,.0f}) vs "
                           f"Q{trim} {annee_precedente} (${ventes_precedente:,.0f}). "
                           f"Investigation recommended."),
                })
            elif variation_yoy >= seuil:
                anomalies.append({
                    "niveau": "INFO",
                    "type": "hausse_yoy",
                    "annee": annee_actuelle, "trimestre": trim,
                    "variation": variation_yoy,
                    "ventes": ventes_actuelle, "ventes_ref": ventes_precedente,
                    "annee_ref": annee_precedente,
                    "fr": (f"INFO - Hausse exceptionnelle de +{variation_yoy:.1f}% au "
                           f"T{trim} {annee_actuelle} ({ventes_actuelle:,.0f} USD) vs "
                           f"T{trim} {annee_precedente} ({ventes_precedente:,.0f} USD)."),
                    "en": (f"INFO - Exceptional +{variation_yoy:.1f}% YoY surge in "
                           f"Q{trim} {annee_actuelle} (${ventes_actuelle:,.0f}) vs "
                           f"Q{trim} {annee_precedente} (${ventes_precedente:,.0f})."),
                })

    anomalies.sort(key=lambda x: (x["niveau"] != "ALERTE", -abs(x["variation"])))
    logger.info(
        f"Anomalies YoY detectees : "
        f"{len([a for a in anomalies if a['niveau'] == 'ALERTE'])} alertes, "
        f"{len([a for a in anomalies if a['niveau'] == 'INFO'])} infos (seuil {seuil}%)"
    )
    return anomalies


def _detecter_anomalie_principale(faits: dict) -> dict:
    """Renvoie le bullet de l'anomalie principale ou du trimestre le moins dynamique."""
    anomalies = detecter_anomalies(faits)
    if anomalies:
        a = anomalies[0]
        return {"fr": a["fr"], "en": a["en"]}

    pt = faits.get("pire_trim") or {}
    variation = pt.get("variation")

    if variation is None or abs(variation) < 0.01:
        variations = faits.get("variations", []) or []
        if variations:
            non_zero = [v for v in variations
                        if v.get("variation") is not None and abs(v["variation"]) >= 0.01]
            if non_zero:
                pt = min(non_zero, key=lambda x: x["variation"])
                variation = pt["variation"]
            else:
                pt = min(variations, key=lambda x: x.get("ventes", 0))
                ventes = pt.get("ventes", 0)
                return {
                    "fr": (f"Trimestre le plus faible en activite : "
                           f"T{pt.get('trimestre', '?')} {pt.get('annee', '?')} "
                           f"({ventes:,.0f} USD). "
                           f"Variation QoQ non calculable sur ce perimetre."),
                    "en": (f"Lowest-activity quarter: "
                           f"Q{pt.get('trimestre', '?')} {pt.get('annee', '?')} "
                           f"(${ventes:,.0f}). "
                           f"QoQ variation not computable on this scope."),
                }

    if variation is not None and variation >= 0:
        return {
            "fr": f"Trimestre le moins dynamique : T{pt.get('trimestre','?')} {pt.get('annee','?')} "
                  f"(croissance QoQ limitee a {variation:+.1f}%).",
            "en": f"Least dynamic quarter: Q{pt.get('trimestre','?')} {pt.get('annee','?')} "
                  f"(QoQ growth limited to {variation:+.1f}%).",
        }

    return {
        "fr": f"Trimestre le plus difficile : T{pt.get('trimestre','?')} {pt.get('annee','?')} "
              f"({variation:+.1f}%).",
        "en": f"Weakest quarter: Q{pt.get('trimestre','?')} {pt.get('annee','?')} "
              f"({variation:+.1f}%).",
    }


def calculer_score(faits: dict) -> dict:
    """Score de performance business /100 sur 4 criteres."""
    score = 0
    details = {}

    cg = faits.get("croissance_globale", 0) or 0
    pts_cg = min(SCORE_WEIGHTS["croissance"], max(0, int(cg / 2)))
    score += pts_cg
    details["croissance"] = {
        "valeur": cg, "points": pts_cg, "max": SCORE_WEIGHTS["croissance"],
        "label": f"Croissance {cg:+.1f}%",
    }

    variations = [abs(v["variation"]) for v in faits["variations"] if v["variation"] is not None]
    if variations:
        variance_moy = sum(variations) / len(variations)
        pts_reg = max(0, SCORE_WEIGHTS["regularite"] - int(variance_moy / 3))
    else:
        variance_moy = 0
        pts_reg = 0
    score += pts_reg
    details["regularite"] = {
        "valeur": round(variance_moy, 1), "points": pts_reg,
        "max": SCORE_WEIGHTS["regularite"], "label": "Regularite trimestrielle",
    }

    mt_var = faits["meilleur_trim"].get("variation")
    if mt_var is None:
        pts_mt = 0
        mt_label = "Meilleur trimestre N/A (variation QoQ non calculable)"
        mt_value_for_details = None
    else:
        pts_mt = min(SCORE_WEIGHTS["meilleur_trim"], max(0, int(mt_var / 2)))
        mt_label = f"Meilleur trimestre +{mt_var:.1f}%"
        mt_value_for_details = mt_var
    score += pts_mt
    details["meilleur_trim"] = {
        "valeur": mt_value_for_details, "points": pts_mt,
        "max": SCORE_WEIGHTS["meilleur_trim"],
        "label": mt_label,
    }

    top3_total = sum(p["ventes"] for p in faits["top_produits"])
    top1_share = faits["top_produits"][0]["ventes"] / top3_total if top3_total else 1
    pts_div = int((1 - top1_share) * 40)
    pts_div = min(SCORE_WEIGHTS["diversite"], max(0, pts_div))
    score += pts_div
    details["diversite"] = {
        "valeur": round(top1_share * 100, 1), "points": pts_div,
        "max": SCORE_WEIGHTS["diversite"],
        "label": f"Diversite produits (top1 = {top1_share * 100:.0f}%)",
    }

    score = min(100, score)
    if score >= 85:
        mention = "Excellent"
    elif score >= 70:
        mention = "Tres bien"
    elif score >= 55:
        mention = "Bien"
    elif score >= 40:
        mention = "Moyen"
    else:
        mention = "A ameliorer"

    return {"score": score, "mention": mention, "details": details}


def afficher_rapport_complet(rapport, resume, anomalies, score_data, langue="fr",
                              structure=None, score_nlp=None):
    """Imprime le rapport complet dans la console."""
    sep = "=" * 65
    print(f"\n{sep}\n   RAPPORT GENERE PAR MISTRAL API\n{sep}\n")
    print(rapport)
    print(f"\n{sep}\n   {resume['titre']}\n{sep}")
    for i, b in enumerate(resume["bullets"], 1):
        print(f"\n  {i}. {b}")
    print(f"\n{sep}\n   SCORE DE PERFORMANCE BUSINESS\n{sep}")
    print(f"\n  Score global : {score_data['score']} / 100  -  {score_data['mention']}\n")
    for k, d in score_data["details"].items():
        bar = "#" * d["points"] + "." * (d["max"] - d["points"])
        print(f"  {d['label']:<40} {bar}  {d['points']}/{d['max']}")

    if score_nlp and score_nlp.get("score_nlp") is not None:
        print(f"\n{sep}\n   SCORE QUALITE NLP DU RAPPORT (Chantier 3)\n{sep}")
        print(f"\n  Score : {score_nlp['score_nlp']}/100  -  {score_nlp.get('mention', '')}\n")
        libelles = {
            "couverture_faits":         "Couverture des faits cles",
            "ancrage_numerique":        "Ancrage numerique",
            "presence_recommandations": "Presence de recommandations",
            "clarte_lisibilite":        "Clarte / lisibilite",
            "absence_repetition":       "Absence de repetition",
            "ton_business":             "Ton business adapte",
        }
        for cle, d in (score_nlp.get("details") or {}).items():
            try:
                if isinstance(d, dict):
                    pts = d.get("points") or d.get("score") or d.get("valeur") or 0
                    maxi = d.get("max") or d.get("max_score") or d.get("poids") or 0
                else:
                    pts, maxi = int(d) if d else 0, 0
                pts = int(pts) if pts is not None else 0
                maxi = int(maxi) if maxi else 0
                if maxi > 0:
                    bar = "#" * pts + "." * max(0, maxi - pts)
                    print(f"  {libelles.get(cle, cle):<32} {bar}  {pts}/{maxi}")
                else:
                    print(f"  {libelles.get(cle, cle):<32} {pts}")
            except Exception:
                print(f"  {libelles.get(cle, cle):<32} {d}")
        if score_nlp.get("lacunes"):
            print(f"\n  Suggestions :")
            for l in score_nlp["lacunes"]:
                print(f"     - {l}")

    if anomalies:
        print(f"\n{sep}\n   ANOMALIES DETECTEES\n{sep}")
        for a in anomalies:
            print(f"\n  {a['fr'] if langue == 'fr' else a['en']}")

    if structure and structure.get("recommendations"):
        print(f"\n{sep}\n   RECOMMANDATIONS HIERARCHISEES (JSON)\n{sep}")
        for i, r in enumerate(structure["recommendations"], 1):
            prio = r.get("priorite", "?").upper()
            niv = r.get("niveau", "?")
            conf = r.get("confiance", "?")
            print(f"\n  [{prio}] [{niv}] [confiance:{conf}]")
            print(f"  -> {r.get('action', '')}")
            if r.get("justification"):
                print(f"     Justification : {r['justification']}")
    print(f"\n{sep}\n")


def generer_tout(faits: dict, langue: str = "fr") -> dict:
    """Pipeline NLP complet : rapport + structure + score + analyse + score NLP."""
    logger.info(f"Generation NLP complete ({langue.upper()})...")

    resultats_2etapes = generer_rapport_avec_structure(faits, langue=langue)
    rapport = resultats_2etapes["rapport"]
    structure = resultats_2etapes["structure"]

    resume = generer_resume_executif(faits, langue=langue)
    anomalies = detecter_anomalies(faits)
    score = calculer_score(faits)

    nltk_rapport = {}
    try:
        nltk_rapport = analyser_rapport_genere(rapport, langue=langue)
        logger.info(
            f"[OK] Analyse NLTK du rapport : {nltk_rapport.get('nb_mots', 0)} mots, "
            f"sentiment={nltk_rapport.get('sentiment', {}).get('compound', 0):+.2f}"
        )
        if "couverture" in nltk_rapport and nltk_rapport["couverture"]:
            cov = nltk_rapport["couverture"]
            cov_score = cov.get("score", cov.get("score_couverture", 0))
            cov_mention = cov.get("mention", "")
            logger.info(f"Couverture business : {cov_score}/100 - {cov_mention}")
    except Exception as e:
        logger.warning(f"Analyse NLTK du rapport echouee: {e}")

    score_nlp = {}
    try:
        from nlp_quality_score import evaluer_qualite_rapport
        compteurs = nltk_rapport.get("compteurs") if nltk_rapport else None
        score_nlp = evaluer_qualite_rapport(
            rapport_texte=rapport,
            faits=faits,
            classifications=compteurs,
            langue=langue,
        )
        logger.info(f"Score qualite NLP : {score_nlp['score_nlp']}/100 - {score_nlp['mention']}")
    except Exception as e:
        logger.warning(f"Calcul du score NLP echoue: {e}")
        score_nlp = {"score_nlp": 0, "mention": "Non evalue", "details": {}, "lacunes": []}

    logger.info(f"[OK] Score performance: {score['score']}/100 - {score['mention']}")
    return {
        "rapport": rapport,
        "structure": structure,
        "resume": resume,
        "anomalies": anomalies,
        "score": score,
        "score_nlp": score_nlp,
        "langue": langue,
        "nltk_rapport": nltk_rapport,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NLP Transformers - Generation de rapport (Mistral API)")
    parser.add_argument("--langue", choices=["fr", "en"], default="fr")
    args = parser.parse_args()

    try:
        spark = create_spark_session()
        df = load_data(spark)
        kpis = compute_kpis(df)
        faits = extraire_faits(kpis)
        spark.stop()
        resultats = generer_tout(faits, langue=args.langue)
        afficher_rapport_complet(
            resultats["rapport"], resultats["resume"],
            resultats["anomalies"], resultats["score"],
            langue=args.langue,
            structure=resultats.get("structure"),
            score_nlp=resultats.get("score_nlp"),
        )
        print("[OK] Etape 4 - nlp_transformers.py terminee")
        print(f"   Langue: {args.langue.upper()}")
        print(f"   Score business: {resultats['score']['score']}/100 - {resultats['score']['mention']}")
        if resultats.get("score_nlp"):
            sn = resultats["score_nlp"]
            print(f"   Score qualite NLP: {sn.get('score_nlp', 0)}/100 - {sn.get('mention', '')}")
    except Exception as e:
        logger.error(f"[KO] Erreur: {e}")
        raise
