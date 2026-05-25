"""
nlp_quality_score.py - Score qualite NLP du rapport (Chantier 3)
==================================================================

Role
----
Evalue la qualite d'un rapport business genere par Mistral selon 6
criteres ponderes. Permet de RENDRE MESURABLE la qualite de la
generation NLP : on peut comparer 2 runs, detecter une regression de
prompt, ou identifier qu'un texte est fluide mais vide.

Pourquoi ce module existe
-------------------------
Un LLM peut produire un texte fluide mais vide (sans chiffres, sans
recommandations concretes, avec du jargon repetitif). Ce module
quantifie ces axes pour en faire une note /100. La note est persistee
en BDD (rapports_nlp.score_nlp) et affichee dans le PDF et l'API.

Bareme (total = 100 points)
---------------------------
1. Couverture des faits cles        (25 pts) - meilleure annee, pire trimestre,
                                              top produit, croissance globale
2. Ancrage numerique                 (25 pts) - densite de chiffres/percentages/montants
3. Presence de recommandations       (15 pts) - verbes d'action ou classifs Chantier 2
4. Clarte / lisibilite               (15 pts) - longueur moyenne phrase + ratio mots uniques
5. Absence de repetition             (10 pts) - diversite des bigrammes
6. Ton business adapte               (10 pts) - vocabulaire metier

Compatibilite avec le Chantier 2
--------------------------------
Si on passe les classifications du Chantier 2 (sous forme de dict
de compteurs OU sous forme de liste de phrases classifiees),
_critere_recommandations() les utilise pour mesurer la presence
de recommandations de maniere plus fiable que la detection de verbes.
Les 2 formats sont supportes (defense en profondeur).

Autonomie
---------
Le module n'utilise PAS NLTK pour ses calculs (split simple par regex).
Il fonctionne donc meme si NLTK n'est pas installe. Utile aussi
comme tests unitaires legers.

Fonction publique principale
----------------------------
    evaluer_qualite_rapport(rapport_texte, faits, classifications=None,
                            langue="fr") -> {
        "score_nlp": int (0-100),
        "mention": str,
        "details": {...},
        "lacunes": [str, ...],
    }
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


VERBES_ACTION_FR = {
    "renforcer", "investir", "developper", "lancer", "automatiser", "diversifier",
    "capitaliser", "stimuler", "reduire", "augmenter", "anticiper", "mettre",
    "deployer", "optimiser", "consolider", "accelerer", "structurer", "piloter",
    "suivre", "ameliorer", "prioriser", "cibler", "definir", "construire",
    "recommander", "proposer", "envisager", "planifier",
}

VERBES_ACTION_EN = {
    "strengthen", "invest", "develop", "launch", "automate", "diversify",
    "capitalize", "stimulate", "reduce", "increase", "anticipate", "implement",
    "deploy", "optimize", "consolidate", "accelerate", "structure", "pilot",
    "monitor", "improve", "prioritize", "target", "define", "build",
    "recommend", "propose", "consider", "plan", "should", "must",
}

TERMES_BUSINESS_FR = {
    "chiffre d'affaires", "croissance", "marge", "rentabilite", "performance",
    "ventes", "commande", "client", "segment", "region", "categorie", "produit",
    "trimestre", "annee", "kpi", "strategie", "marche", "tendance", "saisonnalite",
    "investissement", "opportunite", "risque", "recommandation",
}

TERMES_BUSINESS_EN = {
    "revenue", "growth", "margin", "profitability", "performance",
    "sales", "order", "customer", "segment", "region", "category", "product",
    "quarter", "year", "kpi", "strategy", "market", "trend", "seasonality",
    "investment", "opportunity", "risk", "recommendation",
}


def _split_sentences(texte: str) -> list:
    """Decoupe simple en phrases (autonome, ne depend pas de NLTK)."""
    if not texte:
        return []
    phrases = re.split(r"(?<=[\.!?])\s+(?=[A-ZÀ-ÖØ-Þ])", texte.strip())
    return [p.strip() for p in phrases if len(p.strip()) >= 5]


def _split_words(texte: str) -> list:
    """Tokenisation mots simple."""
    if not texte:
        return []
    return re.findall(r"\b[a-zA-Zà-öø-ÿÀ-ÖØ-Þ]+\b", texte.lower())


def _mention(score: float) -> str:
    """Mention qualitative."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Tres bon"
    if score >= 55:
        return "Bon"
    if score >= 40:
        return "Moyen"
    return "Insuffisant"


def _critere_couverture_faits(rapport_texte: str, faits: dict) -> dict:
    """Critere 1 (25 pts) : meilleure annee, trimestres, top produit, croissance."""
    if not rapport_texte or not faits:
        return {"score": 0, "max": 25, "details": {}, "manquants": ["Rapport ou faits absents"]}

    texte_lower = rapport_texte.lower()
    points = 0
    details = {}
    manquants = []

    ba = faits.get("meilleure_annee", {})
    if ba and ba.get("annee"):
        annee = str(ba["annee"])
        if annee in rapport_texte:
            points += 5
            details["meilleure_annee_mentionnee"] = True
        else:
            details["meilleure_annee_mentionnee"] = False
            manquants.append(f"L'annee record ({annee}) n'est pas mentionnee explicitement.")

    mt = faits.get("meilleur_trim", {})
    if mt and mt.get("trimestre") and mt.get("annee"):
        patterns = [
            f"q{mt['trimestre']} {mt['annee']}",
            f"t{mt['trimestre']} {mt['annee']}",
            f"trimestre {mt['trimestre']} {mt['annee']}",
            f"quarter {mt['trimestre']} {mt['annee']}",
        ]
        if any(p in texte_lower for p in patterns):
            points += 5
            details["meilleur_trim_mentionne"] = True
        else:
            details["meilleur_trim_mentionne"] = False
            manquants.append(f"Le meilleur trimestre (T{mt['trimestre']} {mt['annee']}) n'est pas cite.")

    pt = faits.get("pire_trim", {})
    if pt and pt.get("trimestre") and pt.get("annee"):
        patterns = [
            f"q{pt['trimestre']} {pt['annee']}",
            f"t{pt['trimestre']} {pt['annee']}",
            f"trimestre {pt['trimestre']} {pt['annee']}",
            f"quarter {pt['trimestre']} {pt['annee']}",
        ]
        if any(p in texte_lower for p in patterns):
            points += 5
            details["pire_trim_mentionne"] = True
        else:
            details["pire_trim_mentionne"] = False
            manquants.append(f"Le trimestre faible (T{pt['trimestre']} {pt['annee']}) n'est pas cite.")

    top_prods = faits.get("top_produits", [])
    if top_prods:
        top1 = top_prods[0].get("nom", "")
        if top1 and top1.lower() in texte_lower:
            points += 5
            details["top_produit_mentionne"] = True
        else:
            details["top_produit_mentionne"] = False
            manquants.append(f"Le top produit ({top1}) n'est pas mentionne.")

    crois = faits.get("croissance_globale")
    if crois is not None:
        try:
            cible = round(float(crois))
            valeurs_acceptees = {
                str(cible), str(cible - 1), str(cible + 1),
                str(cible - 2), str(cible + 2),
            }
            pcts_dans_texte = re.findall(r"([+-]?\d+(?:[\.,]\d+)?)\s*%", rapport_texte)
            pcts_int = {str(int(round(float(p.replace(",", "."))))) for p in pcts_dans_texte}
            if valeurs_acceptees & pcts_int:
                points += 5
                details["croissance_mentionnee"] = True
            else:
                details["croissance_mentionnee"] = False
                manquants.append(f"Le taux de croissance globale ({cible}%) n'est pas cite.")
        except (ValueError, TypeError):
            pass

    return {"score": points, "max": 25, "details": details, "manquants": manquants}


def _critere_ancrage_numerique(rapport_texte: str) -> dict:
    """Critere 2 (25 pts) : densite de chiffres dans le texte."""
    if not rapport_texte:
        return {"score": 0, "max": 25, "details": {}, "manquants": ["Rapport vide"]}

    nb_mots = len(_split_words(rapport_texte))
    if nb_mots == 0:
        return {"score": 0, "max": 25, "details": {}, "manquants": ["Aucun mot"]}

    nb_pourcentages = len(re.findall(r"[+-]?\d+(?:[\.,]\d+)?\s*%", rapport_texte))
    nb_montants = len(re.findall(r"[\$€]\s*\d[\d\.,\s]*|\d[\d\.,\s]*\s*(?:USD|EUR|usd|eur)", rapport_texte))
    nb_nombres_purs = len(re.findall(r"\b\d{2,}(?:[\.,]\d+)?\b", rapport_texte))
    nb_total_chiffres = nb_pourcentages + nb_montants + nb_nombres_purs
    densite = (nb_total_chiffres / nb_mots) * 100

    if 4 <= densite <= 12:
        score = 25
    elif 2 <= densite < 4:
        score = 18
    elif 12 < densite <= 18:
        score = 22
    elif 1 <= densite < 2:
        score = 10
    elif densite > 18:
        score = 15
    else:
        score = 3

    manquants = []
    if nb_pourcentages == 0:
        manquants.append("Aucun pourcentage : les variations chiffrees ne sont pas explicitees.")
    if nb_montants == 0 and nb_nombres_purs < 3:
        manquants.append("Trop peu de montants ou de chiffres absolus : ancrage faible.")

    return {
        "score": score,
        "max": 25,
        "details": {
            "nb_mots": nb_mots,
            "nb_pourcentages": nb_pourcentages,
            "nb_montants": nb_montants,
            "nb_nombres_purs": nb_nombres_purs,
            "densite_pour_100_mots": round(densite, 2),
        },
        "manquants": manquants,
    }


def _critere_recommandations(rapport_texte: str, classifications, langue: str) -> dict:
    """
    Critere 3 (15 pts) : presence de recommandations actionnables.

    Accepte 2 formats pour `classifications` :
      - dict de compteurs : {"recommandation": 3, ...}
      - liste plate : [{"intent": "recommandation", "sentence": "..."}, ...]
    Sinon, fallback sur la detection de verbes d'action.
    """
    manquants = []
    details = {}

    nb_recos = None
    nb_total = 0
    if classifications:
        if isinstance(classifications, dict):
            nb_recos = int(classifications.get("recommandation", 0) or 0)
            nb_total = sum(v for v in classifications.values() if isinstance(v, int))
        elif isinstance(classifications, list):
            nb_recos = sum(
                1 for c in classifications
                if isinstance(c, dict) and c.get("intent") == "recommandation"
            )
            nb_total = sum(1 for c in classifications if isinstance(c, dict))

    if nb_recos is not None:
        details["source"] = "chantier_2"
        details["nb_recommandations"] = nb_recos
        details["nb_phrases_classifiees"] = nb_total

        if nb_recos >= 3:
            score = 15
        elif nb_recos == 2:
            score = 12
        elif nb_recos == 1:
            score = 7
        else:
            score = 2
            manquants.append("Aucune phrase classee 'recommandation' par le classifieur (Chantier 2).")
    else:
        verbes = VERBES_ACTION_FR if langue == "fr" else VERBES_ACTION_EN
        mots = _split_words(rapport_texte)
        verbes_trouves = [v for v in verbes if v in mots]
        nb_verbes_uniques = len(set(verbes_trouves))
        details["source"] = "autonome"
        details["nb_verbes_action_uniques"] = nb_verbes_uniques
        details["verbes_detectes"] = verbes_trouves[:8]

        if nb_verbes_uniques >= 5:
            score = 15
        elif nb_verbes_uniques >= 3:
            score = 11
        elif nb_verbes_uniques >= 1:
            score = 6
        else:
            score = 1
            manquants.append("Aucun verbe d'action detecte : pas de recommandation explicite.")

    return {"score": score, "max": 15, "details": details, "manquants": manquants}


def _critere_clarte(rapport_texte: str) -> dict:
    """Critere 4 (15 pts) : longueur moyenne phrase + richesse lexicale."""
    phrases = _split_sentences(rapport_texte)
    mots = _split_words(rapport_texte)
    if not phrases or not mots:
        return {"score": 0, "max": 15, "details": {}, "manquants": ["Texte vide"]}

    nb_phrases = len(phrases)
    nb_mots = len(mots)
    longueur_moy = nb_mots / nb_phrases
    ratio_unique = len(set(mots)) / nb_mots

    if 12 <= longueur_moy <= 22:
        score_longueur = 8
    elif 9 <= longueur_moy < 12 or 22 < longueur_moy <= 28:
        score_longueur = 6
    elif longueur_moy < 9:
        score_longueur = 3
    else:
        score_longueur = 4

    if ratio_unique >= 0.55:
        score_richesse = 7
    elif ratio_unique >= 0.45:
        score_richesse = 6
    elif ratio_unique >= 0.35:
        score_richesse = 4
    else:
        score_richesse = 2

    score_total = score_longueur + score_richesse

    manquants = []
    if longueur_moy > 30:
        manquants.append(f"Phrases trop longues (moyenne : {longueur_moy:.1f} mots) - perte de lisibilite.")
    if longueur_moy < 8:
        manquants.append(f"Phrases trop courtes (moyenne : {longueur_moy:.1f} mots) - style hache.")
    if ratio_unique < 0.35:
        manquants.append(f"Vocabulaire pauvre (TTR={ratio_unique:.2f}) - beaucoup de repetitions.")

    return {
        "score": score_total,
        "max": 15,
        "details": {
            "nb_phrases": nb_phrases,
            "nb_mots": nb_mots,
            "longueur_moy_phrase": round(longueur_moy, 1),
            "ratio_mots_uniques": round(ratio_unique, 3),
        },
        "manquants": manquants,
    }


def _critere_repetition(rapport_texte: str) -> dict:
    """Critere 5 (10 pts) : detection bigrammes repetes."""
    mots = _split_words(rapport_texte)
    if len(mots) < 10:
        return {"score": 0, "max": 10, "details": {}, "manquants": ["Texte trop court"]}

    bigrammes = [(mots[i], mots[i + 1]) for i in range(len(mots) - 1)]
    if not bigrammes:
        return {"score": 5, "max": 10, "details": {}, "manquants": []}

    bigrammes_uniques = len(set(bigrammes))
    ratio_bigrammes = bigrammes_uniques / len(bigrammes)

    from collections import Counter
    compteur = Counter(bigrammes)
    repetitions_excessives = sum(1 for _, n in compteur.items() if n >= 4)

    if ratio_bigrammes >= 0.85 and repetitions_excessives == 0:
        score = 10
    elif ratio_bigrammes >= 0.75 and repetitions_excessives <= 1:
        score = 8
    elif ratio_bigrammes >= 0.65:
        score = 6
    elif ratio_bigrammes >= 0.55:
        score = 4
    else:
        score = 2

    manquants = []
    if repetitions_excessives >= 2:
        manquants.append(f"{repetitions_excessives} expressions repetees 4+ fois - style monotone.")

    return {
        "score": score,
        "max": 10,
        "details": {
            "ratio_bigrammes_uniques": round(ratio_bigrammes, 3),
            "repetitions_excessives": repetitions_excessives,
        },
        "manquants": manquants,
    }


def _critere_ton_business(rapport_texte: str, langue: str) -> dict:
    """Critere 6 (10 pts) : vocabulaire metier."""
    if not rapport_texte:
        return {"score": 0, "max": 10, "details": {}, "manquants": ["Texte vide"]}

    termes = TERMES_BUSINESS_FR if langue == "fr" else TERMES_BUSINESS_EN
    texte_lower = rapport_texte.lower()
    termes_trouves = [t for t in termes if t in texte_lower]
    nb_termes = len(termes_trouves)

    if nb_termes >= 10:
        score = 10
    elif nb_termes >= 7:
        score = 8
    elif nb_termes >= 4:
        score = 6
    elif nb_termes >= 2:
        score = 4
    else:
        score = 1

    manquants = []
    if nb_termes < 4:
        manquants.append(
            f"Vocabulaire business limite ({nb_termes} termes metier) - registre trop generique."
        )

    return {
        "score": score,
        "max": 10,
        "details": {
            "nb_termes_business": nb_termes,
            "exemples": termes_trouves[:6],
        },
        "manquants": manquants,
    }


def evaluer_qualite_rapport(rapport_texte: str, faits: dict,
                             classifications: Optional[object] = None,
                             langue: str = "fr") -> dict:
    """Evaluation complete. Renvoie score, mention, details des 6 criteres, lacunes."""
    if not rapport_texte or not isinstance(rapport_texte, str):
        return {
            "score_nlp": 0,
            "mention": "Non evalue",
            "details": {},
            "lacunes": ["Rapport texte absent ou invalide."],
        }

    if not isinstance(faits, dict):
        faits = {}

    c1 = _critere_couverture_faits(rapport_texte, faits)
    c2 = _critere_ancrage_numerique(rapport_texte)
    c3 = _critere_recommandations(rapport_texte, classifications, langue)
    c4 = _critere_clarte(rapport_texte)
    c5 = _critere_repetition(rapport_texte)
    c6 = _critere_ton_business(rapport_texte, langue)

    score_total = c1["score"] + c2["score"] + c3["score"] + c4["score"] + c5["score"] + c6["score"]
    score_total = max(0, min(100, int(round(score_total))))

    lacunes = []
    for crit in (c1, c2, c3, c4, c5, c6):
        lacunes.extend(crit.get("manquants", []))

    resultat = {
        "score_nlp": score_total,
        "mention": _mention(score_total),
        "details": {
            "couverture_faits": c1,
            "ancrage_numerique": c2,
            "recommandations": c3,
            "clarte": c4,
            "repetition": c5,
            "ton_business": c6,
        },
        "lacunes": lacunes,
    }

    logger.info(
        f"Score qualite NLP : {score_total}/100 - {_mention(score_total)} "
        f"(couv={c1['score']}/25, num={c2['score']}/25, "
        f"reco={c3['score']}/15, clar={c4['score']}/15, "
        f"rep={c5['score']}/10, ton={c6['score']}/10)"
    )
    return resultat


def afficher_score_qualite(resultat: dict) -> None:
    """Imprime le resultat de l'evaluation."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("   SCORE QUALITE NLP DU RAPPORT (Chantier 3)")
    print(sep)
    print(f"   Score global : {resultat['score_nlp']}/100 - {resultat['mention']}")
    print("\n   Detail par critere :")

    libelles = {
        "couverture_faits": "1. Couverture des faits cles",
        "ancrage_numerique": "2. Ancrage numerique       ",
        "recommandations": "3. Recommandations         ",
        "clarte": "4. Clarte / lisibilite     ",
        "repetition": "5. Absence de repetition   ",
        "ton_business": "6. Ton business            ",
    }
    for cle, lib in libelles.items():
        c = resultat["details"].get(cle, {})
        print(f"      {lib} : {c.get('score', 0):>2}/{c.get('max', 0)}")

    if resultat.get("lacunes"):
        print(f"\n   Suggestions d'amelioration ({len(resultat['lacunes'])}) :")
        for lac in resultat["lacunes"][:5]:
            print(f"      - {lac}")
    print(sep + "\n")


if __name__ == "__main__":
    rapport_test = (
        "Sur la periode 2015-2018, le chiffre d'affaires a progresse de 47% "
        "pour atteindre 2.3M USD. L'annee 2018 constitue le meilleur exercice "
        "avec une performance record. Le trimestre Q4 2018 a genere 278 mille "
        "USD, porte par la categorie Technology qui represente 37% du mix. "
        "A l'inverse, le T1 2018 a marque une faiblesse a -47%, confirmant "
        "un pattern de baisse saisonniere recurrent. Les sous-categories Phones "
        "et Chairs dominent le top des ventes. Pour capitaliser sur ces resultats, "
        "il est recommande de renforcer les investissements sur la region West, "
        "de diversifier le portefeuille produits et d'automatiser le suivi des KPI."
    )
    faits_test = {
        "meilleure_annee": {"annee": 2018, "ca": 2300000},
        "meilleur_trim": {"trimestre": 4, "annee": 2018, "ventes": 278000, "variation": 43.7},
        "pire_trim": {"trimestre": 1, "annee": 2018, "ventes": 122000, "variation": -47.8},
        "top_produits": [{"nom": "Phones", "ventes": 100000}],
        "croissance_globale": 47.0,
        "periode": "2015-2018",
    }
    res = evaluer_qualite_rapport(rapport_test, faits_test, langue="fr")
    afficher_score_qualite(res)
