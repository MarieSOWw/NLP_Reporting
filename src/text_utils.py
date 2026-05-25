"""
text_utils.py - Utilitaires de manipulation de texte
=====================================================

Role
----
Module utilitaire centralise, partage par les autres composants du
projet (api.py, generate_pdf.py, nlp_transformers.py). Existe pour
eviter la duplication de code et garantir un comportement uniforme.

Avant la centralisation, quatre copies de _strip_markdown coexistaient
dans le projet, avec de legeres divergences. Le module fournit la
version unique a importer.

Fonctions exposees
------------------
- strip_markdown(text)    : retire tout markdown injecte par un LLM
                            (gras, italique, titres, listes, fences code,
                            separateurs Unicode et ASCII)
- clean_nlp_dict(data)    : applique strip_markdown sur tous les champs
                            textuels d'un dict de rapport NLP en une passe
- couleur_pour_score(s)   : renvoie un code couleur hex selon un score
                            sur 100 (vert / orange / rouge)

L'alias `_strip_markdown` (avec underscore initial) est conserve pour
ne pas casser les imports historiques de fichiers qui n'ont pas encore
migre vers le nom public.
"""

import re
from typing import Optional


def strip_markdown(text: Optional[str]) -> str:
    """
    Supprime tout markdown et formats structurels qu'un LLM peut injecter :
    titres (#), gras (**), italique (*), separateurs (---, ___, ===),
    puces, numerotation, code inline, blocs de code triple backtick,
    titres en MAJUSCULES isoles, prefixes Analyse:/Tendances:, fleches ->.
    Retourne toujours une chaine (jamais None).
    """
    if not text:
        return ""
    t = str(text)

    # Titres markdown
    t = re.sub(r"^#{1,6}\s*.*?$", "", t, flags=re.MULTILINE)
    t = re.sub(r"#{1,6}\s*", "", t)

    # Gras / italique
    t = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", t)
    t = re.sub(r"__(.+?)__", r"\1", t)

    # Separateurs horizontaux (ASCII et Unicode box-drawing)
    t = re.sub(r"^\s*[-—_=═─]{3,}\s*.*?$", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*-{3,}\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s-{3,}\s", " ", t)

    # Listes a puces et numerotees
    t = re.sub(r"^\s*[\-\*••]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)

    # Code inline et blocs
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"```[\s\S]*?```", "", t)

    # Titres en MAJUSCULES isoles
    t = re.sub(
        r"^\s*[A-ZÉÈÀÂÎÔÛÇ]{4,}"
        r"(?:\s+[A-ZÉÈÀÂÎÔÛÇ0-9]{2,}){0,4}\s+",
        "",
        t,
        flags=re.MULTILINE,
    )

    # Prefixes typiques
    t = re.sub(r"^\s*(?:Analyse|Tendances|Opportunite|Note)\s*:\s*", "", t, flags=re.MULTILINE)

    # Fleches emdash
    t = re.sub(r"\s*→\s*", ". ", t)

    # Doubles espaces et triples newlines
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Marges
    t = t.lstrip("\n ").rstrip()
    return t


# Alias retrocompatible
_strip_markdown = strip_markdown


def clean_nlp_dict(data: dict) -> dict:
    """Nettoie tous les champs textuels d'un rapport NLP en une seule passe."""
    if not data:
        return data
    out = dict(data)
    for k in ("rapport_complet", "resume_bullet1", "resume_bullet2", "resume_bullet3"):
        if out.get(k):
            out[k] = strip_markdown(str(out[k]))
    return out


def couleur_pour_score(score: int) -> str:
    """Code couleur hex selon un score sur 100."""
    if score >= 85:
        return "#16a34a"
    if score >= 70:
        return "#22c55e"
    if score >= 55:
        return "#eab308"
    if score >= 40:
        return "#f97316"
    return "#ef4444"
