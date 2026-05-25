"""
business_rules.py - Moteur de recommandations base sur des regles
===================================================================

Role
----
Genere des recommandations business conditionnelles a partir des
faits KPI deja calcules. Complementaire au Chantier 4 (recos LLM
Mistral) : permet de garantir qu'au moins certaines actions
essentielles sont toujours emises, meme si le LLM est indisponible
ou produit du contenu generique.

Pourquoi un moteur a regles
---------------------------
- Reproductible : meme entree -> meme sortie (testable).
- Explicable : chaque reco cite la regle activee.
- Robuste : pas de dependance Mistral, pas de hallucination possible.

Regles implementees
-------------------
R1. Croissance globale < 0%        -> strategie defensive, plan de relance
R2. Region dominante > 35% du CA   -> risque de concentration regionale
R3. Categorie dominante > 35% CA   -> risque de dependance produit
R4. Trimestre en baisse < -10% YoY -> action corrective sur ce trimestre
R5. Panier moyen baisse mais commandes augmentent -> strategie upsell
R6. Volatilite trimestrielle > 30% -> stabilisation operationnelle
R7. Signaux YoY positifs >= 2      -> capitaliser sur les periodes fortes

Format des recommandations
--------------------------
Chaque reco renvoyee contient :
    action          : str
    priorite        : haute | moyenne | basse
    niveau          : strategique | tactique | operationnel
    confiance       : elevee | moyenne | faible
    justification   : str (cite le KPI declencheur)
    kpi_source      : str (nom du KPI utilise)
    regle           : str (identifiant R1-R7)
"""

from typing import List, Dict


def evaluer_regles(faits: dict, langue: str = "fr") -> List[Dict]:
    """
    Applique les 7 regles business au dict `faits`.
    Renvoie la liste des recommandations activees, triees par priorite.
    """
    recos: List[Dict] = []

    cg = faits.get("croissance_globale")
    annuel = faits.get("annuel", []) or []
    variations = faits.get("variations", []) or []
    vol = faits.get("volatilite", {}) or {}
    segments = faits.get("segments", {}) or {}
    top_prods = faits.get("top_produits", []) or []

    # CA total agrege (pour calculer parts)
    ca_total = sum(a.get("ca", 0) or 0 for a in annuel) if annuel else 0

    # R1 - Croissance negative
    if cg is not None and cg < 0:
        if langue == "fr":
            action = (
                f"Activer une strategie defensive : revue des couts, "
                f"plan de relance commercial sur les zones de baisse."
            )
            just = f"Croissance globale negative ({cg:+.1f}%) sur la periode."
        else:
            action = (
                "Activate a defensive strategy: cost review and "
                "commercial recovery plan on declining areas."
            )
            just = f"Negative global growth ({cg:+.1f}%) over the period."
        recos.append({
            "action": action,
            "priorite": "haute",
            "niveau": "strategique",
            "confiance": "elevee",
            "justification": just,
            "kpi_source": "croissance_globale",
            "regle": "R1",
        })

    # R2 - Region dominante > 35%
    # On reconstruit la part des regions depuis les ventes_regions si dispo.
    region_focus = faits.get("region_focus")
    if not region_focus and faits.get("regions"):
        # faits["regions"] est un dict {annee: {region, ventes}}.
        # On agrege par region.
        region_totals = {}
        for info in faits["regions"].values():
            reg = info.get("region")
            if reg:
                region_totals[reg] = region_totals.get(reg, 0) + (info.get("ventes") or 0)
        if region_totals:
            total_regions = sum(region_totals.values()) or 1
            top_region, top_ventes = max(region_totals.items(), key=lambda x: x[1])
            part = top_ventes / total_regions * 100
            if part > 35:
                if langue == "fr":
                    action = (
                        f"Diversifier geographiquement : la region {top_region} "
                        f"represente {part:.1f}% du CA. Developper des relais "
                        f"de croissance dans les autres regions."
                    )
                    just = f"Region {top_region} = {part:.1f}% du CA total."
                else:
                    action = (
                        f"Diversify geographically: {top_region} accounts for "
                        f"{part:.1f}% of revenue. Develop growth relays in "
                        f"other regions."
                    )
                    just = f"Region {top_region} = {part:.1f}% of total revenue."
                recos.append({
                    "action": action,
                    "priorite": "haute" if part > 50 else "moyenne",
                    "niveau": "strategique",
                    "confiance": "elevee",
                    "justification": just,
                    "kpi_source": "ventes_par_region",
                    "regle": "R2",
                })

    # R3 - Categorie / sous-categorie dominante > 35%
    if top_prods:
        total_top = sum(p.get("ventes", 0) for p in top_prods)
        if total_top > 0:
            top1 = top_prods[0]
            part = (top1.get("ventes", 0) / total_top) * 100
            if part > 35:
                if langue == "fr":
                    action = (
                        f"Reduire la dependance produit : {top1['nom']} concentre "
                        f"{part:.1f}% du top 3 des ventes. Renforcer les "
                        f"sous-categories complementaires."
                    )
                    just = f"{top1['nom']} = {part:.1f}% du top 3 sub-categories."
                else:
                    action = (
                        f"Reduce product dependence: {top1['nom']} accounts for "
                        f"{part:.1f}% of top 3 sales. Strengthen complementary "
                        f"sub-categories."
                    )
                    just = f"{top1['nom']} = {part:.1f}% of top 3 sub-categories."
                recos.append({
                    "action": action,
                    "priorite": "moyenne",
                    "niveau": "strategique",
                    "confiance": "elevee",
                    "justification": just,
                    "kpi_source": "top_produits",
                    "regle": "R3",
                })

    # R4 - Trimestre en baisse forte (variation QoQ < -10%)
    trim_baisses = [v for v in variations
                    if v.get("variation") is not None and v["variation"] < -10]
    if trim_baisses:
        pire = min(trim_baisses, key=lambda x: x["variation"])
        if langue == "fr":
            action = (
                f"Action corrective sur le T{pire['trimestre']} {pire['annee']} : "
                f"diagnostic des causes (saisonnalite, marche, ops) et plan "
                f"d'attenuation."
            )
            just = f"T{pire['trimestre']} {pire['annee']} : {pire['variation']:+.1f}% QoQ ({pire['ventes']:,.0f} USD)."
        else:
            action = (
                f"Corrective action on Q{pire['trimestre']} {pire['annee']}: "
                f"root-cause analysis (seasonality, market, ops) and mitigation plan."
            )
            just = f"Q{pire['trimestre']} {pire['annee']}: {pire['variation']:+.1f}% QoQ (${pire['ventes']:,.0f})."
        recos.append({
            "action": action,
            "priorite": "haute" if pire["variation"] < -25 else "moyenne",
            "niveau": "tactique",
            "confiance": "moyenne",
            "justification": just,
            "kpi_source": "kpi_variation",
            "regle": "R4",
        })

    # R5 - Panier moyen baisse mais commandes augmentent (signal upsell/cross-sell)
    if len(annuel) >= 2:
        dernier = annuel[-1]
        avant = annuel[-2]
        panier_d = dernier.get("panier_moyen") or 0
        panier_a = avant.get("panier_moyen") or 0
        cmd_d = dernier.get("commandes") or 0
        cmd_a = avant.get("commandes") or 0
        if panier_a > 0 and cmd_a > 0:
            delta_panier = (panier_d - panier_a) / panier_a * 100
            delta_cmd = (cmd_d - cmd_a) / cmd_a * 100
            if delta_panier < -2 and delta_cmd > 2:
                if langue == "fr":
                    action = (
                        f"Lancer une strategie d'upsell / cross-sell : volume de "
                        f"commandes en hausse ({delta_cmd:+.1f}%) mais panier moyen "
                        f"en baisse ({delta_panier:+.1f}%) entre {avant['annee']} "
                        f"et {dernier['annee']}."
                    )
                    just = (
                        f"Panier {panier_a:.0f}->{panier_d:.0f} ({delta_panier:+.1f}%), "
                        f"commandes {cmd_a:,}->{cmd_d:,} ({delta_cmd:+.1f}%)."
                    )
                else:
                    action = (
                        f"Launch upsell/cross-sell strategy: order volume up "
                        f"({delta_cmd:+.1f}%) but average basket down "
                        f"({delta_panier:+.1f}%) between {avant['annee']} "
                        f"and {dernier['annee']}."
                    )
                    just = (
                        f"Basket {panier_a:.0f}->{panier_d:.0f} ({delta_panier:+.1f}%), "
                        f"orders {cmd_a:,}->{cmd_d:,} ({delta_cmd:+.1f}%)."
                    )
                recos.append({
                    "action": action,
                    "priorite": "moyenne",
                    "niveau": "tactique",
                    "confiance": "elevee",
                    "justification": just,
                    "kpi_source": "kpi_annuel",
                    "regle": "R5",
                })

    # R6 - Volatilite elevee
    vol_val = vol.get("valeur", 0) or 0
    if vol_val > 30:
        if langue == "fr":
            action = (
                f"Mettre en place un dispositif de stabilisation operationnelle "
                f"(buffers stocks, planning glissant) face a une volatilite "
                f"trimestrielle de {vol_val:.1f}%."
            )
            just = f"Volatilite trimestrielle {vol_val:.1f}% (seuil critique 30%)."
        else:
            action = (
                f"Set up an operational stabilization mechanism (inventory "
                f"buffers, rolling planning) given quarterly volatility of "
                f"{vol_val:.1f}%."
            )
            just = f"Quarterly volatility {vol_val:.1f}% (critical threshold 30%)."
        recos.append({
            "action": action,
            "priorite": "haute",
            "niveau": "operationnel",
            "confiance": "elevee",
            "justification": just,
            "kpi_source": "volatilite",
            "regle": "R6",
        })

    # R7 - Signaux YoY positifs >= 2 (basé sur croissance_yoy stockees)
    yoy_positives = [a for a in annuel
                     if a.get("croissance_yoy") is not None and a["croissance_yoy"] > 0]
    if len(yoy_positives) >= 2:
        meilleures = sorted(yoy_positives, key=lambda x: -x["croissance_yoy"])[:2]
        annees_str = ", ".join(str(a["annee"]) for a in meilleures)
        if langue == "fr":
            action = (
                f"Capitaliser sur les periodes fortes ({annees_str}) en "
                f"reproduisant les leviers : campagnes saisonnieres, mix produit, "
                f"focus client gagnant."
            )
            just = (
                f"{len(yoy_positives)} annees en croissance positive, dont "
                + ", ".join(f"{a['annee']} ({a['croissance_yoy']:+.1f}%)" for a in meilleures)
                + "."
            )
        else:
            action = (
                f"Capitalize on strong periods ({annees_str}) by replicating "
                f"levers: seasonal campaigns, product mix, winning customer focus."
            )
            just = (
                f"{len(yoy_positives)} years with positive YoY growth, including "
                + ", ".join(f"{a['annee']} ({a['croissance_yoy']:+.1f}%)" for a in meilleures)
                + "."
            )
        recos.append({
            "action": action,
            "priorite": "moyenne",
            "niveau": "tactique",
            "confiance": "elevee",
            "justification": just,
            "kpi_source": "kpi_annuel.croissance_yoy",
            "regle": "R7",
        })

    # Tri par priorite (haute > moyenne > basse), puis par regle
    ordre_prio = {"haute": 0, "moyenne": 1, "basse": 2}
    recos.sort(key=lambda r: (ordre_prio.get(r["priorite"], 99), r["regle"]))
    return recos


def fusionner_avec_recos_llm(recos_llm: list, recos_regles: list, max_total: int = 7) -> list:
    """
    Fusionne les recos LLM (Chantier 4) avec les recos a regles.
    Strategie : prendre TOUTES les recos a regles haute priorite + completer
    avec les recos LLM jusqu'a max_total.
    """
    if not isinstance(recos_llm, list):
        recos_llm = []
    if not isinstance(recos_regles, list):
        recos_regles = []

    # Recos regles prioritaires (haute uniquement)
    hautes_regles = [r for r in recos_regles if r.get("priorite") == "haute"]
    autres_regles = [r for r in recos_regles if r.get("priorite") != "haute"]

    fusionnees = list(hautes_regles)
    for r in recos_llm:
        if len(fusionnees) >= max_total:
            break
        fusionnees.append(r)
    for r in autres_regles:
        if len(fusionnees) >= max_total:
            break
        fusionnees.append(r)

    # Retri final par priorite
    ordre_prio = {"haute": 0, "moyenne": 1, "basse": 2}
    fusionnees.sort(key=lambda r: ordre_prio.get(r.get("priorite", "moyenne"), 99))
    return fusionnees[:max_total]


if __name__ == "__main__":
    import json
    faits_demo = {
        "croissance_globale": 49.5,
        "annuel": [
            {"annee": 2015, "ca": 477478, "commandes": 934, "panier_moyen": 511.22, "croissance_yoy": None},
            {"annee": 2016, "ca": 453394, "commandes": 1002, "panier_moyen": 452.49, "croissance_yoy": -5.04},
            {"annee": 2017, "ca": 592327, "commandes": 1272, "panier_moyen": 465.67, "croissance_yoy": 30.64},
            {"annee": 2018, "ca": 713926, "commandes": 1634, "panier_moyen": 436.92, "croissance_yoy": 20.53},
        ],
        "variations": [
            {"annee": 2016, "trimestre": 1, "variation": -64.9, "ventes": 62137},
        ],
        "volatilite": {"niveau": "Tres elevee", "valeur": 40.8},
        "top_produits": [
            {"nom": "Phones", "ventes": 327528},
            {"nom": "Chairs", "ventes": 322823},
            {"nom": "Storage", "ventes": 212303},
        ],
        "regions": {
            2018: {"region": "West", "ventes": 244213},
            2017: {"region": "West", "ventes": 178831},
            2016: {"region": "East", "ventes": 152145},
            2015: {"region": "West", "ventes": 144728},
        },
    }
    recos = evaluer_regles(faits_demo, langue="fr")
    print(json.dumps(recos, indent=2, ensure_ascii=False))
