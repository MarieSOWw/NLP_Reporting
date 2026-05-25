"""
pdf_design_professional.py - Systeme de design editorial pour les PDFs
========================================================================

Role
----
Definit la palette de couleurs, l'echelle d'espacement et la
typographie utilisees par generate_pdf.py. Approche editoriale
inspiree de The Economist, McKinsey Quarterly, Bain Reports.

Pourquoi un module dedie
------------------------
- Centralise les choix esthetiques : changer toute l'identite visuelle
  du PDF se fait en un seul endroit.
- Permet de tomber sur les styles legacy de _styles() (dans
  generate_pdf.py) si ce module est absent.

Principes design
----------------
- Hierarchie typographique stricte (4 niveaux maximum).
- Palette monochromatique avec un seul accent (bleu nuit).
- Espacements generaux disciplines (golden ratio).
- Filets fins comme separateurs, pas de bordures lourdes.
- Pas d'images stock, pas d'illustrations decoratives.

Exposes
-------
- ColorPalette       : classe portant tous les codes hex
- MPL_PALETTE        : palette matplotlib coherente avec l'identite
- Spacing            : echelle d'espacement
- _styles_professional() : dict complet de ParagraphStyle
"""

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.units import cm, mm


class ColorPalette:
    """Palette editoriale : neutres profonds + accent unique bleu nuit."""

    INK_BLACK = colors.HexColor("#0E1116")
    INK_DARK = colors.HexColor("#1F2937")
    INK_MID = colors.HexColor("#4B5563")
    INK_SOFT = colors.HexColor("#9CA3AF")

    PRIMARY_BLUE = colors.HexColor("#1E3A5F")
    PRIMARY_DARK = colors.HexColor("#0F2440")
    PRIMARY_LIGHT = colors.HexColor("#E8EEF5")
    PRIMARY_TINT = colors.HexColor("#F4F7FB")

    SUCCESS = colors.HexColor("#15803D")
    WARNING = colors.HexColor("#B45309")
    DANGER = colors.HexColor("#991B1B")
    NEUTRAL = colors.HexColor("#6B7280")

    PAPER = colors.HexColor("#FFFFFF")
    PAPER_TINT = colors.HexColor("#FAFBFC")
    RULE = colors.HexColor("#D1D5DB")
    RULE_LIGHT = colors.HexColor("#E5E7EB")

    # Compatibilite ascendante
    PRIMARY = PRIMARY_BLUE
    GREY_LIGHT = PRIMARY_TINT
    TEXT_DARK = INK_BLACK
    NEUTRAL_GREY = INK_MID
    WHITE = PAPER
    SUCCESS_GREEN = SUCCESS
    WARNING_RED = DANGER
    ACCENT_ORANGE = WARNING


# Palette matplotlib coherente avec l'identite
MPL_PALETTE = [
    "#1E3A5F", "#475569", "#92400E", "#15803D",
    "#7C2D12", "#4338CA", "#0F766E", "#9F1239",
]


class Spacing:
    """Echelle d'espacement disciplinee."""
    XXS = 2
    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32
    XXXL = 48

    SECTION_BEFORE = 28
    SECTION_AFTER = 14
    PARA_AFTER = 8
    CHART_AFTER = 18
    BLOCK_AFTER = 14

    MARGIN_LEFT = 2.2
    MARGIN_RIGHT = 2.2
    MARGIN_TOP = 2.0
    MARGIN_BOTTOM = 2.4


def _styles_professional():
    """Dictionnaire complet de ParagraphStyle utilises par generate_pdf.py."""
    base = getSampleStyleSheet()
    C = ColorPalette

    return {
        "cover_eyebrow": ParagraphStyle(
            "cover_eyebrow", parent=base["Normal"],
            fontSize=10, textColor=C.PRIMARY_LIGHT, alignment=TA_LEFT,
            fontName="Helvetica-Bold", leading=14, spaceAfter=8,
        ),
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"],
            fontSize=42, textColor=C.PAPER, alignment=TA_LEFT,
            fontName="Helvetica-Bold", leading=46, spaceAfter=10,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"],
            fontSize=14, textColor=C.PRIMARY_LIGHT, alignment=TA_LEFT,
            leading=20, spaceAfter=6, fontName="Helvetica",
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", parent=base["Normal"],
            fontSize=9, textColor=C.PRIMARY_LIGHT, alignment=TA_LEFT,
            leading=14, spaceAfter=4, fontName="Helvetica",
        ),

        "section_eyebrow": ParagraphStyle(
            "section_eyebrow", parent=base["Normal"],
            fontSize=8, textColor=C.NEUTRAL, alignment=TA_LEFT,
            fontName="Helvetica-Bold", leading=10, spaceBefore=0, spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontSize=22, textColor=C.INK_BLACK,
            spaceBefore=10, spaceAfter=14,
            fontName="Helvetica-Bold", leading=26,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontSize=14, textColor=C.PRIMARY_BLUE,
            spaceBefore=18, spaceAfter=8,
            fontName="Helvetica-Bold", leading=18,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontSize=11, textColor=C.INK_DARK,
            spaceBefore=12, spaceAfter=6,
            fontName="Helvetica-Bold", leading=14,
        ),

        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10.5, textColor=C.INK_DARK, leading=17,
            spaceAfter=8, alignment=TA_JUSTIFY, fontName="Helvetica",
        ),
        "body_lead": ParagraphStyle(
            "body_lead", parent=base["Normal"],
            fontSize=12, textColor=C.INK_BLACK, leading=19,
            spaceAfter=10, alignment=TA_JUSTIFY, fontName="Helvetica",
        ),
        "body_small": ParagraphStyle(
            "body_small", parent=base["Normal"],
            fontSize=9, textColor=C.INK_MID, leading=14,
            spaceAfter=6, fontName="Helvetica",
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"],
            fontSize=8.5, textColor=C.INK_MID, leading=12,
            spaceBefore=2, spaceAfter=12, alignment=TA_LEFT,
            fontName="Helvetica-Oblique",
        ),

        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"],
            fontSize=10.5, textColor=C.INK_DARK, leading=16,
            leftIndent=18, bulletIndent=6, spaceAfter=6, fontName="Helvetica",
        ),

        "kpi_val": ParagraphStyle(
            "kpi_val", parent=base["Normal"],
            fontSize=22, textColor=C.INK_BLACK, alignment=TA_CENTER,
            fontName="Helvetica-Bold", leading=26, spaceAfter=2,
        ),
        "kpi_lbl": ParagraphStyle(
            "kpi_lbl", parent=base["Normal"],
            fontSize=8, textColor=C.NEUTRAL, alignment=TA_CENTER,
            fontName="Helvetica-Bold", leading=11,
        ),

        "interp": ParagraphStyle(
            "interp", parent=base["Normal"],
            fontSize=9.5, textColor=C.INK_MID, leading=14,
            spaceBefore=4, spaceAfter=12, leftIndent=14, rightIndent=4,
            fontName="Helvetica-Oblique",
        ),

        "alerte": ParagraphStyle(
            "alerte", parent=base["Normal"],
            fontSize=10, textColor=C.DANGER, leading=15,
            spaceAfter=6, fontName="Helvetica-Bold",
        ),
        "info": ParagraphStyle(
            "info", parent=base["Normal"],
            fontSize=10, textColor=C.SUCCESS, leading=15,
            spaceAfter=6, fontName="Helvetica-Bold",
        ),
        "warning": ParagraphStyle(
            "warning", parent=base["Normal"],
            fontSize=10, textColor=C.WARNING, leading=15,
            spaceAfter=6, fontName="Helvetica-Bold",
        ),

        "reco": ParagraphStyle(
            "reco", parent=base["Normal"],
            fontSize=10.5, textColor=C.INK_DARK, leading=16,
            leftIndent=20, spaceAfter=8, fontName="Helvetica",
        ),

        "footer": ParagraphStyle(
            "footer", parent=base["Normal"],
            fontSize=8, textColor=C.INK_SOFT, alignment=TA_CENTER,
            fontName="Helvetica", leading=11,
        ),
        "methodo": ParagraphStyle(
            "methodo", parent=base["Normal"],
            fontSize=9.5, textColor=C.INK_DARK, leading=15,
            spaceAfter=4, fontName="Helvetica",
        ),
    }


def get_color_palette(theme="default"):
    """Retourne la palette en dict (compat ascendante)."""
    C = ColorPalette
    return {
        "primary": C.PRIMARY_BLUE,
        "primary_dark": C.PRIMARY_DARK,
        "primary_light": C.PRIMARY_LIGHT,
        "accent": C.WARNING,
        "success": C.SUCCESS,
        "warning": C.DANGER,
    }


_styles = _styles_professional


if __name__ == "__main__":
    styles = _styles_professional()
    print("Systeme de design 'Editorial Business' charge.")
    print(f"{len(styles)} styles disponibles :")
    for name in sorted(styles.keys()):
        print(f"   - {name}")
