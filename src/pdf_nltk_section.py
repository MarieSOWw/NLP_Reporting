"""
pdf_nltk_section.py - Section "Analyse Linguistique NLTK" du PDF
==================================================================

Role
----
Construit la section dediee a l'analyse linguistique dans le PDF
business. Affiche les resultats de l'analyse NLTK du rapport
genere par Mistral :
    - Statistiques textuelles (nb phrases, mots, mots uniques)
    - Top mots-cles extraits
    - Tonalite business (lexique metier, alternative a VADER)
    - Themes business dominants
    - Entites nommees (NER)
    - Echantillon de POS tags

Pourquoi ce module separe
-------------------------
generate_pdf.py est deja tres long. Cette section a une logique
specifique (tableaux, formats heterogenes) qui merite son module.
Elle est appelee depuis generate_pdf via `creer_section_analyse_linguistique`.

Gestion des formats heterogenes
-------------------------------
Les `keywords` et `themes` peuvent arriver sous 3 formats selon la
source de donnees :
    - [("mot", 7), ...]               - tuples natifs (live)
    - ["mot1", "mot2", ...]            - strings (analyser_rapport_genere)
    - "mot1:7, mot2:5"                  - CSV depuis la table nltk_analysis
Les helpers _normaliser_keywords et _normaliser_themes normalisent
tout vers list[tuple[str, int]] pour eviter les erreurs d'unpacking.

API publique
------------
- creer_section_analyse_linguistique(styles, nltk_data, lang="fr")
    Renvoie la liste d'elements ReportLab a ajouter a la story.
"""

from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm


def _normaliser_keywords(raw):
    """Normalise les mots-cles en liste de tuples (mot, freq)."""
    if not raw:
        return []

    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, tuple) and len(item) >= 2:
                try:
                    out.append((str(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, list) and len(item) >= 2:
                try:
                    out.append((str(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, dict):
                mot = item.get("mot") or item.get("word") or item.get("keyword") or ""
                freq = item.get("freq") or item.get("count") or item.get("n") or 1
                if mot:
                    try:
                        out.append((str(mot), int(freq)))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(item, str) and item.strip():
                out.append((item.strip(), 1))
        return out

    if isinstance(raw, str):
        out = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                k, _, v = part.partition(":")
                k = k.strip()
                v = v.strip()
                try:
                    out.append((k, int(v)))
                except (TypeError, ValueError):
                    out.append((k, 1))
            else:
                out.append((part, 1))
        return out

    return []


def _normaliser_themes(raw):
    """Normalise les themes en liste de tuples (theme, score)."""
    if not raw:
        return []

    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, tuple) and len(item) >= 2:
                try:
                    out.append((str(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, list) and len(item) >= 2:
                try:
                    out.append((str(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, dict):
                t = item.get("theme") or item.get("name") or ""
                s = item.get("score") or item.get("count") or 1
                if t:
                    try:
                        out.append((str(t), int(s)))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(item, str) and item.strip():
                out.append((item.strip(), 1))
        return out

    if isinstance(raw, str):
        out = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                k, _, v = part.partition(":")
                try:
                    out.append((k.strip(), int(v.strip())))
                except (TypeError, ValueError):
                    out.append((k.strip(), 1))
            else:
                out.append((part, 1))
        return out

    return []


_LIBELLES_THEMES_FR = {
    "croissance": "Croissance",
    "decroissance": "Decroissance",
    "saisonnalite": "Saisonnalite",
    "concentration_produit": "Concentration produit",
    "concentration_regionale": "Concentration regionale",
    "volatilite": "Volatilite",
    "fidelisation": "Fidelisation client",
    "panier_moyen": "Panier moyen",
    "diversification": "Diversification",
    "risque": "Risques",
    "opportunite": "Opportunites",
    "recommandation": "Recommandations",
}

_LIBELLES_THEMES_EN = {
    "croissance": "Growth",
    "decroissance": "Decline",
    "saisonnalite": "Seasonality",
    "concentration_produit": "Product concentration",
    "concentration_regionale": "Regional concentration",
    "volatilite": "Volatility",
    "fidelisation": "Customer retention",
    "panier_moyen": "Average basket",
    "diversification": "Diversification",
    "risque": "Risks",
    "opportunite": "Opportunities",
    "recommandation": "Recommendations",
}


def creer_section_analyse_linguistique(styles, nltk_data: dict, lang: str = "fr") -> list:
    """Renvoie la liste d'elements ReportLab a ajouter a la story."""
    elements = []
    if not nltk_data:
        return elements

    titre = "ANALYSE LINGUISTIQUE (NLTK)" if lang == "fr" else "LINGUISTIC ANALYSIS (NLTK)"
    elements.append(Paragraph(titre, styles["h1"]))
    elements.append(Spacer(1, 0.3 * cm))

    intro = (
        "Cette section presente les resultats de l'analyse textuelle approfondie "
        "realisee avec NLTK (Natural Language Toolkit) sur le rapport genere, "
        "enrichie d'une evaluation de la tonalite business."
        if lang == "fr" else
        "This section presents the results of in-depth textual analysis "
        "performed with NLTK on the generated report, enriched with "
        "a business-tone evaluation."
    )
    elements.append(Paragraph(intro, styles["body"]))
    elements.append(Spacer(1, 0.5 * cm))

    # 1. Statistiques
    stats = nltk_data.get("stats", {})
    if stats:
        subtitle = "Statistiques textuelles" if lang == "fr" else "Text Statistics"
        elements.append(Paragraph(subtitle, styles["h2"]))

        data_stats = [
            ["Metrique" if lang == "fr" else "Metric",
             "Valeur" if lang == "fr" else "Value"],
            ["Nombre de phrases" if lang == "fr" else "Number of sentences",
             str(stats.get("nb_phrases", 0))],
            ["Nombre de mots" if lang == "fr" else "Number of words",
             str(stats.get("nb_mots", 0))],
            ["Mots uniques" if lang == "fr" else "Unique words",
             str(stats.get("mots_uniques", 0))],
            ["Longueur moyenne phrase" if lang == "fr" else "Avg sentence length",
             f"{stats.get('longueur_moyenne_phrase', 0):.1f} mots"],
        ]

        table_stats = Table(data_stats, colWidths=[10 * cm, 5 * cm])
        table_stats.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(table_stats)
        elements.append(Spacer(1, 0.5 * cm))

    # 2. Mots-cles
    keywords_normalises = _normaliser_keywords(nltk_data.get("keywords"))
    if keywords_normalises:
        subtitle = "Mots-cles extraits (TF-IDF)" if lang == "fr" else "Extracted Keywords (TF-IDF)"
        elements.append(Paragraph(subtitle, styles["h2"]))
        top_kw = keywords_normalises[:5]
        kw_text = ", ".join([f"<b>{mot}</b> ({freq})" for mot, freq in top_kw])
        elements.append(Paragraph(kw_text, styles["body"]))
        elements.append(Spacer(1, 0.5 * cm))

    # 3. Tonalite business
    tonalite = nltk_data.get("tonalite_business")
    if tonalite:
        subtitle = (
            "Tonalite business du rapport"
            if lang == "fr" else
            "Business tone of the report"
        )
        elements.append(Paragraph(subtitle, styles["h2"]))

        explication = (
            "Cette evaluation s'appuie sur un <b>lexique metier</b> "
            "(croissance / opportunites vs decroissance / risques) plutot que "
            "sur VADER, dont les scores de sentiment generalistes sont peu "
            "fiables sur du texte B2B."
            if lang == "fr" else
            "This evaluation relies on a <b>business lexicon</b> "
            "(growth / opportunities vs decline / risks) rather than on "
            "VADER, whose general-purpose sentiment scores are unreliable "
            "on B2B text."
        )
        elements.append(Paragraph(
            explication,
            styles["body_small"] if "body_small" in styles else styles["body"],
        ))
        elements.append(Spacer(1, 0.2 * cm))

        map_bg = {
            "favorable":   colors.HexColor("#d4edda"),
            "neutre":      colors.HexColor("#fff3cd"),
            "defavorable": colors.HexColor("#f8d7da"),
        }
        bg_color = map_bg.get(tonalite.get("tonalite", "neutre"), colors.HexColor("#ecf0f1"))

        data_ton = [
            ["Indicateur" if lang == "fr" else "Indicator",
             "Valeur" if lang == "fr" else "Value"],
            [
                "Tonalite globale" if lang == "fr" else "Overall tone",
                tonalite.get("tonalite_label", tonalite.get("tonalite", "-")),
            ],
            [
                "Score business" if lang == "fr" else "Business score",
                f"{tonalite.get('score_business', 0):+d} / 100",
            ],
            [
                "Niveau de risque percu" if lang == "fr" else "Perceived risk level",
                tonalite.get("niveau_risque_label", tonalite.get("niveau_risque", "-")),
            ],
            [
                "Stabilite du discours" if lang == "fr" else "Discourse stability",
                tonalite.get("stabilite_label", tonalite.get("stabilite", "-")),
            ],
            [
                "Signaux positifs" if lang == "fr" else "Positive signals",
                str(tonalite.get("signaux_positifs", 0)),
            ],
            [
                "Signaux negatifs" if lang == "fr" else "Negative signals",
                str(tonalite.get("signaux_negatifs", 0)),
            ],
        ]

        table_ton = Table(data_ton, colWidths=[9 * cm, 6 * cm])
        table_ton.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
            ("BACKGROUND",  (0, 1), (-1, 1), bg_color),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(table_ton)
        elements.append(Spacer(1, 0.5 * cm))

    else:
        sentiment = nltk_data.get("sentiment", {})
        if sentiment:
            subtitle = (
                "Sentiment du rapport (VADER - legacy)"
                if lang == "fr" else
                "Report sentiment (VADER - legacy)"
            )
            elements.append(Paragraph(subtitle, styles["h2"]))
            elements.append(Paragraph(
                "Score VADER : " + f"{sentiment.get('compound', 0):+.2f}",
                styles["body"],
            ))
            elements.append(Spacer(1, 0.5 * cm))

    # 4. Themes business
    themes_raw = nltk_data.get("themes") or nltk_data.get("themes_business") or []
    themes_normalises = _normaliser_themes(themes_raw)

    if themes_normalises:
        subtitle = (
            "Themes business dominants"
            if lang == "fr" else
            "Dominant business themes"
        )
        elements.append(Paragraph(subtitle, styles["h2"]))

        mapping = _LIBELLES_THEMES_FR if lang == "fr" else _LIBELLES_THEMES_EN
        data_th = [["Theme" if lang == "fr" else "Theme", "Occurrences"]]
        total_occ = sum(sc for _, sc in themes_normalises) or 1
        for theme, score in themes_normalises[:7]:
            libelle = mapping.get(theme, theme.replace("_", " ").capitalize())
            pct = int(round(score / total_occ * 100))
            data_th.append([libelle, f"{score}  ({pct}%)"])

        table_th = Table(data_th, colWidths=[9 * cm, 6 * cm])
        table_th.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f1efe8")]),
        ]))
        elements.append(table_th)
        elements.append(Spacer(1, 0.5 * cm))

    # 5. Entites nommees
    entities = nltk_data.get("entities", {})
    if entities and any(entities.values()):
        subtitle = "Entites nommees detectees (NER)" if lang == "fr" else "Named Entities Detected (NER)"
        elements.append(Paragraph(subtitle, styles["h2"]))

        ent_lines = []
        for ent_type, values in entities.items():
            if values:
                values_str = ", ".join(values[:3])
                if len(values) > 3:
                    values_str += f" (+{len(values) - 3} autres)"
                ent_lines.append(f"<b>{ent_type}</b> : {values_str}")

        if ent_lines:
            ent_text = "<br/>".join(ent_lines)
            elements.append(Paragraph(ent_text, styles["body"]))
        else:
            no_ent = "Aucune entite detectee." if lang == "fr" else "No entities detected."
            elements.append(Paragraph(no_ent, styles["body"]))
        elements.append(Spacer(1, 0.5 * cm))

    # 6. POS tagging echantillon
    pos_tags = nltk_data.get("pos_tags_sample", [])
    if pos_tags:
        subtitle = "Etiquetage grammatical (POS Tags - echantillon)" if lang == "fr" else "Part-of-Speech Tagging (Sample)"
        elements.append(Paragraph(subtitle, styles["h2"]))

        data_pos = [["Mot" if lang == "fr" else "Word", "Tag"]]
        for item in pos_tags[:10]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                data_pos.append([str(item[0]), str(item[1])])
            elif isinstance(item, dict):
                data_pos.append([
                    str(item.get("mot") or item.get("word") or ""),
                    str(item.get("tag") or item.get("pos") or ""),
                ])

        if len(data_pos) > 1:
            table_pos = Table(data_pos, colWidths=[8 * cm, 4 * cm])
            table_pos.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(table_pos)
            elements.append(Spacer(1, 0.5 * cm))

    note = (
        "<i>Note : Cette analyse NLTK combine plusieurs techniques de NLP "
        "(tokenisation, lemmatisation, NER, POS tagging, classification "
        "par intention, tonalite business par lexique metier) "
        "pour extraire des insights linguistiques a partir des donnees.</i>"
        if lang == "fr" else
        "<i>Note: This NLTK analysis combines several NLP techniques "
        "(tokenization, lemmatization, NER, POS tagging, intent "
        "classification, business-tone via domain lexicon) "
        "to extract linguistic insights from the data.</i>"
    )
    elements.append(Paragraph(note, styles["body"]))
    elements.append(Spacer(1, 1 * cm))
    return elements


if __name__ == "__main__":
    print("Test _normaliser_keywords:")
    print("  tuples:", _normaliser_keywords([("ventes", 7), ("clients", 5)]))
    print("  strings:", _normaliser_keywords(["ventes", "clients"]))
    print("  CSV:", _normaliser_keywords("ventes:7, clients:5, segment:3"))
    print("  None:", _normaliser_keywords(None))

    print("\nTest _normaliser_themes:")
    print("  tuples:", _normaliser_themes([("croissance", 7)]))
    print("  strings:", _normaliser_themes(["croissance", "volatilite"]))
    print("  CSV:", _normaliser_themes("croissance, volatilite"))
