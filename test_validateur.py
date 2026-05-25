"""Test du validateur numerique sur les vraies incoherences du PDF du 21/05."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.nlp_transformers import valider_chiffres_dans_texte, _collecter_valeurs_autorisees

# Faits reconstruits a partir du PDF rapport_business_fr_global_20260521_004749.pdf
faits = {
    "periode": "2015-2018",
    "annuel": [
        {"annee": 2015, "ca": 477_000, "commandes": 934,  "panier_moyen": 511, "croissance_yoy": None},
        {"annee": 2016, "ca": 453_000, "commandes": 1002, "panier_moyen": 452, "croissance_yoy": -5.0},
        {"annee": 2017, "ca": 592_000, "commandes": 1272, "panier_moyen": 466, "croissance_yoy": 30.6},
        {"annee": 2018, "ca": 714_000, "commandes": 1634, "panier_moyen": 437, "croissance_yoy": 20.5},
    ],
    "croissance_globale": 49.5,
    "variations": [
        {"annee": 2017, "trimestre": 4, "ventes": 234_389, "variation": 69.8},
        {"annee": 2016, "trimestre": 1, "ventes":  62_358, "variation": -64.9},
        {"annee": 2017, "trimestre": 2, "ventes": 135_061, "variation": 54.0},
    ],
    "top_produits": [
        {"nom": "Phones",  "ventes": 327_782, "qte": 0},
        {"nom": "Chairs",  "ventes": 322_569, "qte": 0},
        {"nom": "Storage", "ventes": 219_000, "qte": 0},
    ],
    "segments": {
        "Consumer":   {"ventes_total": 1_148_061, "commandes": 0},
        "Corporate":  {"ventes_total":   688_000, "commandes": 0},
        "Home Office":{"ventes_total":   425_000, "commandes": 0},
    },
    "volatilite": {"valeur": 40.9, "niveau": "elevee"},
    "regions": {},
    "mensuel": [],
    "meilleure_annee": {"annee": 2018, "ca": 714_000},
    "pire_trim": {"annee": 2016, "trimestre": 1, "variation": -64.9},
}

# ============ TEXTE QUI CONTIENT LES VRAIES INCOHERENCES ============
prose_reelle = """
Sur la periode 2015-2018, le chiffre d'affaires a affiche une progression remarquable
de pres de 50%, passant de 1,1M USD a 1,6M USD, malgre une volatilite trimestrielle
moyenne de 40,8% qui revele une instabilite persistante.

Le quatrieme trimestre 2017 s'est distingue avec une croissance de 69,5% atteignant
230 957 USD, tandis que le premier trimestre 2016 a enregistre une chute de 64,9%
et un chiffre de 62 137 USD.

Le segment Consumer s'impose avec 1,1M USD generes. Les telephones et chaises
representent ensemble 650 351 USD. La croissance annuelle de 20,5% en 2018 par
rapport a 2017 offre une fenetre d'opportunite.

Le chiffre d'affaires record est de 713,927 USD en 2018.
"""

# ============ TEXTE CORRECT POUR COMPARER ============
prose_correcte = """
Sur la periode 2015-2018, l'activite atteint 2.2M USD avec une croissance de 49.5%.
Le quatrieme trimestre 2017 a genere 234,389 USD soit +69.8% QoQ.
Le segment Consumer represente 1,148,061 USD. La volatilite est de 40.9%.
"""

print("=" * 70)
print("TEST 1 : Prose contenant les hallucinations reelles du PDF")
print("=" * 70)
suspects_reels = valider_chiffres_dans_texte(prose_reelle, faits)
print(f"\nNombre de chiffres suspects detectes : {len(suspects_reels)}\n")
for extrait, valeur, raison in suspects_reels:
    print(f"  - '{extrait}'  (valeur={valeur})  -> {raison}")

print()
print("=" * 70)
print("TEST 2 : Prose avec les VRAIS chiffres (ne devrait rien detecter)")
print("=" * 70)
suspects_ok = valider_chiffres_dans_texte(prose_correcte, faits)
print(f"\nNombre de chiffres suspects detectes : {len(suspects_ok)}\n")
for extrait, valeur, raison in suspects_ok:
    print(f"  - '{extrait}'  (valeur={valeur})  -> {raison}")

print()
print("=" * 70)
print("TEST 3 : Justification reco erronee (32.8% du CA 2017)")
print("=" * 70)
justif_erronee = "T4 2017 a genere 234,389 USD, soit 32.8% du chiffre d'affaires annuel 2017."
suspects_justif = valider_chiffres_dans_texte(justif_erronee, faits)
print(f"\nNombre de chiffres suspects : {len(suspects_justif)}\n")
for extrait, valeur, raison in suspects_justif:
    print(f"  - '{extrait}'  (valeur={valeur})  -> {raison}")

print()
print("=" * 70)
print("TEST 4 : Action de reco avec 45.7% du total (vu dans PDF 02:49)")
print("=" * 70)
action_erronee = ("Renforcer la visibilite des produits Phones et Chairs, "
                  "qui representent ensemble 650,605 USD (45.7% du total), "
                  "via des promotions ciblees.")
suspects_action = valider_chiffres_dans_texte(action_erronee, faits)
print(f"\nNombre de chiffres suspects : {len(suspects_action)}\n")
for extrait, valeur, raison in suspects_action:
    print(f"  - '{extrait}'  (valeur={valeur})  -> {raison}")

print()
print("=" * 70)
print("RESULTAT")
print("=" * 70)
if len(suspects_reels) >= 3 and len(suspects_ok) == 0:
    print("[OK] Validateur fonctionne : il attrape les hallucinations sans faux positifs.")
else:
    print(f"[KO] suspects_reels={len(suspects_reels)} (attendu >= 3), "
          f"suspects_ok={len(suspects_ok)} (attendu = 0)")
