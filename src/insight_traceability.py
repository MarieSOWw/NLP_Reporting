"""
insight_traceability.py - Tracabilite insight -> KPI -> confiance
====================================================================

Role
----
Pour chaque insight critique d'un rapport business, produit une trace
qui relie l'insight a sa source de donnees. Permet de justifier au
lecteur (ou a un auditeur) que l'IA ne parle pas au hasard : chaque
phrase est rattachee a un KPI concret avec sa valeur source.

Pourquoi
--------
Un rapport NLP sans tracabilite est une boite noire. Avec la
tracabilite :
- on peut auditer chaque affirmation
- on demontre l'absence d'hallucination
- on quantifie la confiance par insight

Format d'un insight trace
-------------------------
    {
        "type":         "tendance | concentration | anomalie | recommandation | volatilite | segment",
        "insight":      "phrase courte resumant l'insight",
        "kpi_source":   "nom du KPI utilise (kpi_global, ventes_par_region, ...)",
        "evidence":     "chiffres source qui justifient l'insight",
        "confidence":   "elevee | moyenne | faible",
        "justification": "comment on est passe du KPI a l'insight",
    }

Strategie de confiance
----------------------
- elevee : insight base sur un KPI agrege deterministe (CA, croissance YoY)
           avec un ecart significatif (> 20% ou > 35% de part)
- moyenne : insight base sur une comparaison qui peut varier selon le
            perimetre (panier moyen, part regionale)
- faible : insight derive (anomalie sur 1 trimestre, signal faible)
"""

from typing import List, Dict


def construire_traces(faits: dict, langue: str = "fr") -> List[Dict]:
    """
    Construit la liste des insights traces a partir des faits KPI.
    Renvoie 5 a 10 insights, chacun relie a sa source.
    """
    traces: List[Dict] = []

    cg = faits.get("croissance_globale")
    periode = faits.get("periode", "")
    annuel = faits.get("annuel", []) or []
    variations = faits.get("variations", []) or []
    vol = faits.get("volatilite", {}) or {}
    segments = faits.get("segments", {}) or {}
    top_prods = faits.get("top_produits", []) or []
    meilleure_annee = faits.get("meilleure_annee", {}) or {}
    pire_annee = faits.get("pire_annee", {}) or {}
    meilleur_trim = faits.get("meilleur_trim", {}) or {}
    pire_trim = faits.get("pire_trim", {}) or {}

    # Trace 1 : tendance globale
    if cg is not None:
        if langue == "fr":
            sens = "hausse" if cg > 2 else ("baisse" if cg < -2 else "stagnation")
            insight = f"Tendance de fond en {sens} ({cg:+.1f}%) sur {periode}."
            justif = (
                f"Calcule depuis kpi_annuel : (CA_fin - CA_debut) / CA_debut * 100."
            )
        else:
            sens = "growth" if cg > 2 else ("decline" if cg < -2 else "stagnation")
            insight = f"Underlying {sens} trend ({cg:+.1f}%) over {periode}."
            justif = "Computed from kpi_annuel: (CA_end - CA_start) / CA_start * 100."

        confiance = "elevee" if abs(cg) > 20 else ("moyenne" if abs(cg) > 5 else "faible")
        ev = ", ".join(f"{a['annee']}={a['ca']:,.0f}" for a in annuel[:4])
        traces.append({
            "type": "tendance",
            "insight": insight,
            "kpi_source": "kpi_annuel",
            "evidence": ev,
            "confidence": confiance,
            "justification": justif,
        })

    # Trace 2 : meilleure annee
    if meilleure_annee:
        if langue == "fr":
            insight = (
                f"Meilleure annee : {meilleure_annee.get('annee')} "
                f"avec {meilleure_annee.get('ca', 0):,.0f} USD."
            )
            justif = "Max(CA_Annuel) sur kpi_annuel."
        else:
            insight = (
                f"Best year: {meilleure_annee.get('annee')} "
                f"with ${meilleure_annee.get('ca', 0):,.0f}."
            )
            justif = "Max(CA_Annuel) on kpi_annuel."
        traces.append({
            "type": "tendance",
            "insight": insight,
            "kpi_source": "kpi_annuel",
            "evidence": f"CA_max={meilleure_annee.get('ca', 0):,.0f}, annee={meilleure_annee.get('annee')}",
            "confidence": "elevee",
            "justification": justif,
        })

    # Trace 3 : concentration regionale
    if faits.get("regions"):
        region_totals = {}
        for info in faits["regions"].values():
            reg = info.get("region")
            if reg:
                region_totals[reg] = region_totals.get(reg, 0) + (info.get("ventes") or 0)
        if region_totals:
            total = sum(region_totals.values()) or 1
            top_reg, top_v = max(region_totals.items(), key=lambda x: x[1])
            part = top_v / total * 100
            if langue == "fr":
                insight = (
                    f"Region {top_reg} concentre {part:.1f}% des ventes."
                )
                justif = "Calcule depuis la somme des ventes par region (ventes_detail / kpi_region_trim)."
            else:
                insight = f"Region {top_reg} concentrates {part:.1f}% of sales."
                justif = "Computed from sum of sales by region (ventes_detail / kpi_region_trim)."
            confiance = "elevee" if part > 35 else "moyenne"
            ev = ", ".join(f"{r}={v:,.0f}" for r, v in sorted(region_totals.items(), key=lambda x: -x[1]))
            traces.append({
                "type": "concentration",
                "insight": insight,
                "kpi_source": "ventes_par_region",
                "evidence": ev,
                "confidence": confiance,
                "justification": justif,
            })

    # Trace 4 : concentration produit
    if top_prods and len(top_prods) >= 3:
        total = sum(p.get("ventes", 0) for p in top_prods)
        if total > 0:
            top1 = top_prods[0]
            part = top1.get("ventes", 0) / total * 100
            if langue == "fr":
                insight = (
                    f"La sous-categorie {top1['nom']} represente "
                    f"{part:.1f}% des 3 produits leaders."
                )
                justif = "Calcule depuis kpi_top_produits sur la part des top 3."
            else:
                insight = (
                    f"Sub-category {top1['nom']} accounts for "
                    f"{part:.1f}% of top 3 leaders."
                )
                justif = "Computed from kpi_top_produits, share of top 3."
            ev = ", ".join(f"{p['nom']}={p['ventes']:,.0f}" for p in top_prods[:3])
            traces.append({
                "type": "concentration",
                "insight": insight,
                "kpi_source": "kpi_top_produits",
                "evidence": ev,
                "confidence": "elevee" if part > 35 else "moyenne",
                "justification": justif,
            })

    # Trace 5 : trimestre meilleur
    if meilleur_trim and meilleur_trim.get("variation") is not None:
        if langue == "fr":
            insight = (
                f"Meilleur trimestre : T{meilleur_trim['trimestre']} "
                f"{meilleur_trim['annee']} ({meilleur_trim['variation']:+.1f}%)."
            )
            justif = "Max(Variation_Pct) sur kpi_variation."
        else:
            insight = (
                f"Best quarter: Q{meilleur_trim['trimestre']} "
                f"{meilleur_trim['annee']} ({meilleur_trim['variation']:+.1f}%)."
            )
            justif = "Max(Variation_Pct) on kpi_variation."
        traces.append({
            "type": "tendance",
            "insight": insight,
            "kpi_source": "kpi_variation",
            "evidence": (
                f"variation={meilleur_trim['variation']:+.1f}%, "
                f"ventes={meilleur_trim.get('ventes', 0):,.0f}"
            ),
            "confidence": "elevee",
            "justification": justif,
        })

    # Trace 6 : trimestre pire (anomalie)
    if pire_trim and pire_trim.get("variation") is not None and pire_trim["variation"] < -10:
        if langue == "fr":
            insight = (
                f"Anomalie : T{pire_trim['trimestre']} {pire_trim['annee']} "
                f"({pire_trim['variation']:+.1f}%)."
            )
            justif = "Min(Variation_Pct) sur kpi_variation, seuil < -10%."
        else:
            insight = (
                f"Anomaly: Q{pire_trim['trimestre']} {pire_trim['annee']} "
                f"({pire_trim['variation']:+.1f}%)."
            )
            justif = "Min(Variation_Pct) on kpi_variation, threshold < -10%."
        traces.append({
            "type": "anomalie",
            "insight": insight,
            "kpi_source": "kpi_variation",
            "evidence": (
                f"variation={pire_trim['variation']:+.1f}%, "
                f"ventes={pire_trim.get('ventes', 0):,.0f}"
            ),
            "confidence": "elevee" if abs(pire_trim["variation"]) > 25 else "moyenne",
            "justification": justif,
        })

    # Trace 7 : segment leader
    if segments:
        try:
            top_seg, top_seg_info = max(segments.items(), key=lambda x: x[1].get("ventes_total", 0))
            if langue == "fr":
                insight = (
                    f"Segment leader : {top_seg} "
                    f"({top_seg_info.get('ventes_total', 0):,.0f} USD)."
                )
                justif = "Max(SUM(ventes)) groupe par segment sur kpi_segment."
            else:
                insight = (
                    f"Leading segment: {top_seg} "
                    f"(${top_seg_info.get('ventes_total', 0):,.0f})."
                )
                justif = "Max(SUM(sales)) grouped by segment on kpi_segment."
            ev = ", ".join(
                f"{s}={info.get('ventes_total', 0):,.0f}"
                for s, info in segments.items()
            )
            traces.append({
                "type": "segment",
                "insight": insight,
                "kpi_source": "kpi_segment",
                "evidence": ev,
                "confidence": "elevee",
                "justification": justif,
            })
        except Exception:
            pass

    # Trace 8 : volatilite
    vol_val = vol.get("valeur", 0) or 0
    if vol_val > 0:
        if langue == "fr":
            insight = f"Volatilite trimestrielle {vol.get('niveau', '?')} ({vol_val:.1f}%)."
            justif = "Moyenne des |variations QoQ| absolues sur kpi_variation."
        else:
            insight = f"Quarterly volatility {vol.get('niveau', '?')} ({vol_val:.1f}%)."
            justif = "Average of |QoQ variations| absolute on kpi_variation."
        traces.append({
            "type": "volatilite",
            "insight": insight,
            "kpi_source": "kpi_variation",
            "evidence": f"avg(|variation|)={vol_val:.1f}%, nb_trimestres={len(variations)}",
            "confidence": "moyenne",
            "justification": justif,
        })

    return traces


if __name__ == "__main__":
    import json
    faits_demo = {
        "periode": "2015-2018",
        "croissance_globale": 49.5,
        "annuel": [
            {"annee": 2015, "ca": 477478},
            {"annee": 2016, "ca": 453394},
            {"annee": 2017, "ca": 592327},
            {"annee": 2018, "ca": 713926},
        ],
        "meilleure_annee": {"annee": 2018, "ca": 713926},
        "meilleur_trim": {"trimestre": 4, "annee": 2017, "variation": 69.45, "ventes": 230957},
        "pire_trim": {"trimestre": 1, "annee": 2016, "variation": -64.9, "ventes": 62137},
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
        "segments": {
            "Consumer": {"ventes_total": 1137124},
            "Corporate": {"ventes_total": 678949},
            "Home Office": {"ventes_total": 421054},
        },
    }
    traces = construire_traces(faits_demo, langue="fr")
    print(json.dumps(traces, indent=2, ensure_ascii=False))
