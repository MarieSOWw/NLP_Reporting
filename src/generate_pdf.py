"""
generate_pdf.py - Generateur de rapports executifs PDF
========================================================

Role
----
Sixieme et dernier maillon du pipeline. Construit un PDF business
d'environ 25 pages avec couverture editoriale, 6 cartes KPI, 13
graphiques contextuels, anomalies, recommandations hierarchisees,
section NLTK enrichie, score qualite NLP, methodologie et conclusion.

Architecture
------------
Le PDF est genere par ReportLab (engine), avec matplotlib pour les
graphiques. Une fonction principale `generer_pdf()` orchestre la
recuperation des donnees depuis PostgreSQL et l'assemblage de la
story ReportLab.

Strategie de filtrage
---------------------
Chaque PDF est genere pour un report_type donne (global, by_year,
by_region, by_category) avec des filtres optionnels. Les graphiques
s'adaptent au perimetre :
- mono-annee : decomposition trimestrielle au lieu du CA annuel
- mono-region : pas de breakdown par region (trivial)
- mono-categorie : top sous-categories filtrees

Source des donnees
------------------
Toutes les donnees viennent de PostgreSQL via src/db.py. Aucune
valeur n'est hardcodee. Si une table est vide, on tombe sur une
section vide proprement (avec message explicite).

Recuperation NLP intelligente
-----------------------------
_recuperer_nlp_data_pour_pdf gere 3 sources par ordre de priorite :
1. resultats passes en parametre (cas main.py)
2. Cache rapports_nlp en BDD (cas API)
3. Generation live via Mistral (cas filtres inhabituels)

Robustesse aux valeurs None
---------------------------
Les helpers _safe_float, _safe_int et _fmt_growth gerent les valeurs
None pour eviter les crashes sur rapports mono-annee (croissance
non calculable). La regle : None = "non calculable" affiche tel quel,
jamais converti silencieusement en 0.

Design
------
Styles charges depuis src/pdf_design_professional.py (palette
editoriale type McKinsey/Bain). Fallback sur _styles() local si le
module pro est absent.

Sortie
------
PDF dans outputs/, nom horodate :
    rapport_business_{lang}_{report_type}{suffix}_{timestamp}.pdf
"""

import os
import re
import sys
import io
import logging
from pathlib import Path
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIT v5.6 — HELPERS DÉFENSIFS (Bug #1)
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(value, default=0.0):
    """
    Convertit en float en gérant None, "", chaînes vides, valeurs
    invalides. Avant : `float(None)` plantait toute la génération PDF
    pour le rapport 2015 (croissance_globale=None car pas d'année N-1).
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    """Idem _safe_float mais pour les entiers."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt_growth(value, default_text="N/A"):
    """
    Formate une croissance YoY en gérant le cas None (première année).
    Avant : `f"{float(None):+.1f}%"` → crash.
    """
    if value is None:
        return default_text
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return default_text

# ── Configuration des chemins ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ═══════════════════════════════════════════════════════════════════════════
#  v5.7 — NETTOYAGE MARKDOWN (source unique de vérité)
# ═══════════════════════════════════════════════════════════════════════════
#  La fonction _strip_markdown locale a été supprimée. On importe désormais
#  strip_markdown depuis src.text_utils, qui est la version maintenue et
#  partagée avec api.py et nlp_transformers.py.
#  L'alias `_strip_markdown` est conservé pour ne pas casser les ~20 appels
#  existants dans ce fichier.
# ═══════════════════════════════════════════════════════════════════════════

from src.text_utils import strip_markdown as _strip_markdown


# Import section analyse linguistique NLTK
from src.pdf_nltk_section import creer_section_analyse_linguistique

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak, Image, KeepTogether
)

# Import config
try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)

# Import du module de lecture PostgreSQL
from src.db import (
    get_kpi_global, get_kpi_filtered_summary,
    get_kpi_annual, get_kpi_annual_filtered,
    get_kpi_regions_summary,
    get_kpi_categories_summary, get_kpi_quarterly, get_kpi_monthly,
    get_kpi_segments_summary, get_kpi_top_products, get_anomalies,
    get_nlp_report, get_nltk_analysis, get_kpi_regions,
    get_kpi_sub_categories,
    _has_filters,
)

# LIVRAISON — Import module images online (optionnel, fallback si absent)
try:
    from src.pdf_images_online import inject_images_to_pdf
    _IMAGES_ONLINE_AVAILABLE = True
except ImportError:
    _IMAGES_ONLINE_AVAILABLE = False
    def inject_images_to_pdf(story, section_name, st):
        pass  # No-op si module absent

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  PALETTE & STYLES
# ═══════════════════════════════════════════════════════════════════════════

class C:
    """Couleurs du rapport — palette originale restaurée à la demande de l'utilisateur."""
    BLEU = colors.HexColor("#1A5FA8")
    BLEU_CL = colors.HexColor("#E6F1FB")
    BLEU_F = colors.HexColor("#0D3F6E")
    ORANGE = colors.HexColor("#EF9F27")
    ROUGE = colors.HexColor("#E24B4A")
    VERT = colors.HexColor("#639922")
    GRIS = colors.HexColor("#888780")
    GRIS_CL = colors.HexColor("#F1EFE8")
    NOIR = colors.HexColor("#2C2C2A")
    BLANC = colors.white
    # Filets pour le design (couleurs neutres conservées)
    RULE = colors.HexColor("#D1D5DB")
    RULE_LIGHT = colors.HexColor("#E5E7EB")

# Palette graphiques — RESTAURÉE à l'identique de l'ancien rapport
MPL_COLORS = ["#1A5FA8", "#639922", "#EF9F27", "#E24B4A",
              "#9B59B6", "#1ABC9C", "#F39C12", "#2C3E50"]


def _styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle("ct", parent=base["Title"], fontSize=28, textColor=C.BLANC, alignment=TA_CENTER, fontName="Helvetica-Bold", leading=34),
        "cover_sub": ParagraphStyle("cs", parent=base["Normal"], fontSize=14, textColor=colors.HexColor("#C0D6F0"), alignment=TA_CENTER, leading=20),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=16, textColor=C.BLEU, spaceBefore=22, spaceAfter=10, fontName="Helvetica-Bold"),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, textColor=C.BLEU_F, spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold"),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=10, textColor=C.NOIR, leading=16, spaceAfter=6, alignment=TA_JUSTIFY),
        "body_small": ParagraphStyle("bs", parent=base["Normal"], fontSize=9, textColor=C.GRIS, leading=13, spaceAfter=4),
        "bullet": ParagraphStyle("blt", parent=base["Normal"], fontSize=10, textColor=C.NOIR, leading=15, leftIndent=20, spaceAfter=4),
        "kpi_val": ParagraphStyle("kv", parent=base["Normal"], fontSize=22, textColor=C.BLEU, alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "kpi_lbl": ParagraphStyle("kl", parent=base["Normal"], fontSize=9, textColor=C.GRIS, alignment=TA_CENTER),
        "interp": ParagraphStyle("interp", parent=base["Normal"], fontSize=9, textColor=C.BLEU_F, leading=13, spaceAfter=8, leftIndent=10, borderPadding=4),
        "alerte": ParagraphStyle("al", parent=base["Normal"], fontSize=10, textColor=C.ROUGE, leading=14, spaceAfter=4),
        "info": ParagraphStyle("inf", parent=base["Normal"], fontSize=10, textColor=C.VERT, leading=14, spaceAfter=4),
        "reco": ParagraphStyle("reco", parent=base["Normal"], fontSize=10, textColor=C.NOIR, leading=15, leftIndent=20, spaceAfter=6),
        "footer": ParagraphStyle("ft", parent=base["Normal"], fontSize=8, textColor=C.GRIS, alignment=TA_CENTER),
        "methodo": ParagraphStyle("meth", parent=base["Normal"], fontSize=9, textColor=C.NOIR, leading=14, spaceAfter=4),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITAIRES GRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════════

def _fig_to_image(fig, width=16*cm, height=9*cm):
    """Convertit une figure matplotlib en Image ReportLab."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=height)


def _fmt_money(v):
    if v >= 1_000_000:
        return f"${v/1_000_000:,.1f}M"
    elif v >= 1_000:
        return f"${v/1_000:,.0f}k"
    return f"${v:,.0f}"


def _chart_interpretation(story, st, text):
    """Ajoute une interprétation business sous un graphique."""
    story.append(Paragraph(f"<i>{text}</i>", st["interp"]))


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPHIQUES (~15)
# ═══════════════════════════════════════════════════════════════════════════

def _chart_ca_annuel(annual):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    # v5.7 — Forcer années en int (cohérence avec _chart_croissance_yoy)
    years = [int(r["annee"]) for r in annual]
    cas = [float(r["ca_annuel"]) for r in annual]
    bars = ax.bar(years, cas, color=MPL_COLORS[:len(years)], width=0.6, edgecolor="white")
    for bar, v in zip(bars, cas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, _fmt_money(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Chiffre d'Affaires par Année", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("CA (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_money(x)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(years)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    return fig


def _chart_croissance_yoy(annual):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    data = [(r["annee"], float(r["croissance_yoy"])) for r in annual if r["croissance_yoy"] is not None]
    if not data:
        plt.close(fig)
        return None
    years, vals = zip(*data)
    # v5.7 — Forcer les années en int (avant: 2016.0, 2016.5...)
    years = [int(y) for y in years]
    colors_bar = [MPL_COLORS[1] if v >= 0 else MPL_COLORS[3] for v in vals]
    ax.bar(years, vals, color=colors_bar, width=0.6)
    for y, v in zip(years, vals):
        ax.text(y, v + (1 if v >= 0 else -3), f"{v:+.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_title("Croissance Annuelle (YoY)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Variation (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # v5.7 — Axe X en années entières uniquement
    ax.set_xticks(years)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    return fig


def _chart_regions(regions):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = [r["region"] for r in regions]
    vals = [float(r["ventes_totales"]) for r in regions]
    ax.barh(names[::-1], vals[::-1], color=MPL_COLORS[:len(names)], height=0.5)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 2000, i, _fmt_money(v), va="center", fontsize=9)
    ax.set_title("Ventes par Région", fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_money(x)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def _chart_categories(categories):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = [r["category"] for r in categories]
    vals = [float(r["ventes_totales"]) for r in categories]
    wedges, texts, autotexts = ax.pie(vals, labels=names, autopct="%1.1f%%", colors=MPL_COLORS[:len(names)], startangle=90)
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    ax.set_title("Répartition par Catégorie", fontsize=12, fontweight="bold", pad=10)
    return fig


def _chart_quarterly(quarterly):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [f"T{r['trimestre']} {r['annee']}" for r in quarterly]
    # FIX BUG — `if r["variation_pct"]` confondait 0.0 (vraie valeur,
    # ex: stagnation parfaite) avec None (non calculable). Désormais on
    # ne remplace que None.
    vals = [
        float(r["variation_pct"]) if r.get("variation_pct") is not None else 0
        for r in quarterly
    ]
    colors_bar = [MPL_COLORS[1] if v >= 0 else MPL_COLORS[3] for v in vals]
    ax.bar(range(len(vals)), vals, color=colors_bar, width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_title("Variations Trimestrielles (QoQ)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Variation (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_monthly(monthly):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [f"{r['annee']}-{int(r['mois']):02d}" for r in monthly]
    vals = [float(r["ventes_mensuelles"]) for r in monthly]
    ax.plot(range(len(vals)), vals, color=MPL_COLORS[0], linewidth=1.5, marker="o", markersize=2)
    if monthly[0].get("moyenne_mobile_3m"):
        mm3 = [float(r["moyenne_mobile_3m"]) if r["moyenne_mobile_3m"] else None for r in monthly]
        mm3_clean = [(i, v) for i, v in enumerate(mm3) if v is not None]
        if mm3_clean:
            ax.plot([x[0] for x in mm3_clean], [x[1] for x in mm3_clean], color=MPL_COLORS[3], linewidth=1.5, linestyle="--", label="Moy. mobile 3M")
            ax.legend(fontsize=8)
    step = max(1, len(labels) // 12)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=45, ha="right", fontsize=7)
    ax.set_title("Évolution Mensuelle du CA", fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_money(x)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_top_products(products):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["sub_category"] for r in products[:10]]
    vals = [float(r["ventes_totales"]) for r in products[:10]]
    bars = ax.barh(names[::-1], vals[::-1], color=MPL_COLORS[0], height=0.6)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + 1000, bar.get_y() + bar.get_height()/2, _fmt_money(v), va="center", fontsize=8)
    ax.set_title("Top 10 Sous-catégories", fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_segments(segments):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = [r["segment"] for r in segments]
    vals = [float(r["ventes_totales"]) for r in segments]
    wedges, texts, autotexts = ax.pie(vals, labels=names, autopct="%1.1f%%", colors=MPL_COLORS[:len(names)], startangle=90, pctdistance=0.8)
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    ax.set_title("Répartition par Segment Client", fontsize=12, fontweight="bold", pad=10)
    return fig


def _chart_ca_quarterly_abs(quarterly):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [f"T{r['trimestre']} {r['annee']}" for r in quarterly]
    vals = [float(r["ventes_totales"]) for r in quarterly]
    ax.fill_between(range(len(vals)), vals, alpha=0.3, color=MPL_COLORS[0])
    ax.plot(range(len(vals)), vals, color=MPL_COLORS[0], linewidth=2, marker="o", markersize=3)
    step = max(1, len(labels) // 8)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=45, ha="right", fontsize=7)
    ax.set_title("CA Trimestriel (valeur absolue)", fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_money(x)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_region_year(region_data):
    """Graphique régions × années (stacked ou grouped)."""
    from collections import defaultdict
    by_region = defaultdict(lambda: defaultdict(float))
    for r in region_data:
        by_region[r["region"]][r["annee"]] += float(r["ventes_totales"])
    if not by_region:
        return None
    regions = sorted(by_region.keys())
    years = sorted({r["annee"] for r in region_data})
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(years))
    w = 0.8 / len(regions)
    for i, reg in enumerate(regions):
        vals = [by_region[reg].get(y, 0) for y in years]
        offset = (i - len(regions)/2 + 0.5) * w
        ax.bar([xi + offset for xi in x], vals, width=w, label=reg, color=MPL_COLORS[i % len(MPL_COLORS)])
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_title("Performance Régionale par Année", fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_money(x)))
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_commandes_annuelles(annual):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    years = [int(r["annee"]) for r in annual]
    orders = [int(r["nb_commandes"]) for r in annual]
    ax.plot(years, orders, color=MPL_COLORS[1], linewidth=2, marker="s", markersize=6)
    for y, o in zip(years, orders):
        ax.text(y, o + 40, f"{o:,}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Évolution du Nombre de Commandes", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Commandes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(years)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    return fig


def _chart_panier_moyen(annual):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    years = [int(r["annee"]) for r in annual]
    baskets = [float(r["panier_moyen"]) for r in annual]
    ax.bar(years, baskets, color=MPL_COLORS[4], width=0.5)
    for y, b in zip(years, baskets):
        ax.text(y, b + 2, f"${b:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Panier Moyen par Année", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("USD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(years)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    return fig


def _chart_croissance_cumulee(quarterly):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [f"T{r['trimestre']} {r['annee']}" for r in quarterly]
    # FIX BUG — Même cas que _chart_quarterly : on ne remplace que None,
    # pas 0.0 (le premier trimestre a légitimement une croissance cumulée
    # de 0% par construction).
    vals = [
        float(r["croissance_cumul"]) if r.get("croissance_cumul") is not None else 0
        for r in quarterly
    ]
    ax.fill_between(range(len(vals)), vals, alpha=0.2, color=MPL_COLORS[1])
    ax.plot(range(len(vals)), vals, color=MPL_COLORS[1], linewidth=2)
    step = max(1, len(labels) // 8)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=45, ha="right", fontsize=7)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    # AUDIT v5.6 Bug #13 — titre dynamique selon le périmètre filtré
    # (avant : "depuis T1 2015" codé en dur même pour rapports filtrés 2017/2018)
    if quarterly:
        first = quarterly[0]
        titre = f"Croissance Cumulée depuis T{first['trimestre']} {first['annee']}"
    else:
        titre = "Croissance Cumulée"
    ax.set_title(titre, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Croissance (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPHIQUES INTELLIGENTS v5 — adaptés au contexte (filtres actifs)
# ═══════════════════════════════════════════════════════════════════════════

def _chart_quarterly_breakdown(quarterly, year=None):
    """Décomposition trimestrielle pour UNE année."""
    if not quarterly:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [f"T{r['trimestre']}" for r in quarterly]
    vals = [float(r["ventes_totales"]) for r in quarterly]
    bars = ax.bar(labels, vals, color=MPL_COLORS[:len(vals)], width=0.55, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                _fmt_money(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
    title = f"Décomposition Trimestrielle — {year}" if year else "Décomposition Trimestrielle"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("CA (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_money(x)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_sub_categories(sub_cats, context=""):
    """Top sous-catégories."""
    if not sub_cats:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["sub_category"] for r in sub_cats[:10]]
    vals = [float(r["ventes_totales"]) for r in sub_cats[:10]]
    bars = ax.barh(names[::-1], vals[::-1],
                   color=[MPL_COLORS[i % len(MPL_COLORS)] for i in range(len(names))],
                   height=0.6, edgecolor="white")
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                _fmt_money(v), va="center", fontsize=8)
    title = f"Top Sous-catégories{' — ' + context if context else ''}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_money(x)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_segments_breakdown(segments, context=""):
    """Donut chart segments."""
    if not segments:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = [r["segment"] for r in segments]
    vals = [float(r["ventes_totales"]) for r in segments]
    if sum(vals) == 0:
        plt.close(fig)
        return None
    wedges, texts, autotexts = ax.pie(
        vals, labels=names, autopct=lambda p: f"{p:.1f}%\n({_fmt_money(p*sum(vals)/100)})",
        colors=MPL_COLORS[:len(names)], startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white"),
        pctdistance=0.78
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
        t.set_color("white")
    title = f"Mix Segments Clients{' — ' + context if context else ''}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SECTIONS DU RAPPORT
# ═══════════════════════════════════════════════════════════════════════════

def _draw_cover_canvas_on_canvas(canvas, page_w, page_h, report_type, filters,
                                  langue, glob):
    """
    Dessine la couverture directement sur le canvas ReportLab — pleine page.
    
    v6.1 : Ajoute une photo d'entrepôt Superstore en arrière-plan (avec
    overlay sombre pour la lisibilité), des accents colorés (filets et
    badge), et conserve le style sombre demandé par l'utilisateur.
    """
    from reportlab.lib import colors as _colors
    from reportlab.lib.utils import ImageReader

    # ── Fond bleu nuit pleine page (base) ──
    canvas.setFillColor(_colors.HexColor("#0F2440"))
    canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    # ── Photo d'entrepôt en arrière-plan (top 60% de la page) ──
    try:
        from src.pdf_images_online import get_cover_photo_bytes
    except ImportError:
        try:
            from pdf_images_online import get_cover_photo_bytes
        except ImportError:
            get_cover_photo_bytes = lambda: None

    photo_bytes = None
    try:
        photo_bytes = get_cover_photo_bytes()
    except Exception as e:
        logger.warning(f"Photo couverture indisponible: {e}")

    if photo_bytes:
        try:
            import io as _io
            photo_reader = ImageReader(_io.BytesIO(photo_bytes))
            # Photo en haut, occupant ~58% de la hauteur
            photo_h = page_h * 0.58
            canvas.drawImage(
                photo_reader,
                0, page_h - photo_h,
                width=page_w, height=photo_h,
                preserveAspectRatio=False, mask='auto',
            )
            # Overlay dégradé sombre par-dessus (5 bandes de transparence
            # croissante pour simuler un dégradé linear-gradient)
            for i in range(40):
                alpha = 0.25 + (i / 40) * 0.65   # 0.25 → 0.90
                canvas.setFillColorRGB(0.06, 0.14, 0.25, alpha=alpha)
                band_h = photo_h / 40
                canvas.rect(
                    0, page_h - photo_h + i * band_h,
                    page_w, band_h + 0.5,  # +0.5 pour éviter les jointures
                    fill=1, stroke=0,
                )
        except Exception as e:
            logger.warning(f"Image cover non rendue: {e}")

    # ── Bande sombre dense en bas (pour les métadonnées) ──
    canvas.setFillColor(_colors.HexColor("#0F2440"))
    canvas.rect(0, 0, page_w, page_h * 0.42, fill=1, stroke=0)

    # ── Filet supérieur d'identité (accent coloré) ──
    canvas.setStrokeColor(_colors.HexColor("#E8EEF5"))
    canvas.setLineWidth(0.8)
    canvas.line(2.2*cm, page_h - 4*cm, 11*cm, page_h - 4*cm)
    # Accent vert (couleur signature) sur la partie gauche du filet
    canvas.setStrokeColor(_colors.HexColor("#639922"))
    canvas.setLineWidth(2.5)
    canvas.line(2.2*cm, page_h - 4*cm, 5*cm, page_h - 4*cm)

    # ── Eyebrow gauche ──
    type_labels_eyebrow = {
        "global":      "GLOBAL  /  ALL DIMENSIONS",
        "by_year":     f"YEAR  /  {filters.get('year','')}",
        "by_region":   f"REGION  /  {(filters.get('region','') or '').upper()}",
        "by_category": f"CATEGORY  /  {(filters.get('category','') or '').upper()}",
    }
    eyebrow = type_labels_eyebrow.get(report_type, "BUSINESS REPORT")
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(_colors.HexColor("#E8EEF5"))
    canvas.drawString(2.2*cm, page_h - 4.7*cm, eyebrow)

    # ── Numéro de rapport (à droite) ──
    canvas.drawRightString(
        page_w - 2.2*cm, page_h - 4.7*cm,
        f"No.  {datetime.now().strftime('%Y%m%d')}"
    )

    # ── Titre principal monumental (3 lignes) ──
    type_titles = {
        "global":      ("Business", "Performance", "Report"),
        "by_year":     (f"Year {filters.get('year','')}", "Performance", "Review"),
        "by_region":   (f"{filters.get('region','')}", "Regional", "Analysis"),
        "by_category": (f"{filters.get('category','')}", "Category", "Analysis"),
    }
    titre_lignes = type_titles.get(report_type, ("Business", "Report", ""))

    canvas.setFillColor(_colors.HexColor("#FFFFFF"))
    canvas.setFont("Helvetica-Bold", 50)
    # Position des lignes plus haut, dans la zone photo (avec overlay)
    canvas.drawString(2.2*cm, page_h - 11.5*cm, titre_lignes[0])
    canvas.drawString(2.2*cm, page_h - 14.0*cm, titre_lignes[1])
    if titre_lignes[2]:
        canvas.setFillColor(_colors.HexColor("#9CB3D2"))
        canvas.drawString(2.2*cm, page_h - 16.5*cm, titre_lignes[2])

    # ── Filet sous le titre (accent orange = couleur d'action) ──
    canvas.setStrokeColor(_colors.HexColor("#EF9F27"))
    canvas.setLineWidth(2.5)
    canvas.line(2.2*cm, page_h - 17.6*cm, 8.5*cm, page_h - 17.6*cm)

    # ── Sous-titre / dataset ──
    canvas.setFillColor(_colors.HexColor("#FFFFFF"))
    canvas.setFont("Helvetica", 14)
    canvas.drawString(2.2*cm, page_h - 18.6*cm, "Superstore Sales Dataset")

    periode = (glob or {}).get("periode", "—")
    canvas.setFillColor(_colors.HexColor("#9CB3D2"))
    canvas.setFont("Helvetica", 11)
    canvas.drawString(2.2*cm, page_h - 19.4*cm, f"Period — {periode}")

    # ── Filet de séparation (bas de page) ──
    canvas.setStrokeColor(_colors.HexColor("#1A5FA8"))
    canvas.setLineWidth(0.8)
    canvas.line(2.2*cm, 4.5*cm, page_w - 2.2*cm, 4.5*cm)

    # ── 3 colonnes méta ──
    metas = [
        ("GENERATED",
         datetime.now().strftime('%d %B %Y').upper(),
         datetime.now().strftime('%H:%M')),
        ("LANGUAGE",
         "FRENCH" if langue == "fr" else "ENGLISH",
         "AUTOMATED"),
        ("PIPELINE",
         "v6.1",
         "ANALYTICS"),
    ]
    col_x = [2.2*cm, 8.5*cm, 14.5*cm]
    for x_col, (lbl, line1, line2) in zip(col_x, metas):
        canvas.setFillColor(_colors.HexColor("#9CB3D2"))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(x_col, 3.7*cm, lbl)

        canvas.setFillColor(_colors.HexColor("#FFFFFF"))
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(x_col, 2.9*cm, line1)

        canvas.setFillColor(_colors.HexColor("#9CB3D2"))
        canvas.setFont("Helvetica", 8.5)
        canvas.drawString(x_col, 2.3*cm, line2)

    # ── Footer technologique ──
    canvas.setFillColor(_colors.HexColor("#FFFFFF"))
    canvas.setFont("Helvetica-Bold", 9.5)
    canvas.drawString(2.2*cm, 1.4*cm, "NLP Business Reporting Platform")

    canvas.setFillColor(_colors.HexColor("#9CB3D2"))
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(2.2*cm, 0.85*cm,
                      "Spark  \u00b7  NLTK  \u00b7  Mistral Transformer  \u00b7  PostgreSQL  \u00b7  FastAPI")


def _section_cover(story, st, glob, report_type, filters, langue):
    """
    Page de couverture — la couverture elle-même est dessinée par
    `_draw_cover_canvas_on_canvas` via le callback onFirstPage.
    Ici, on n'ajoute qu'un PageBreak pour passer à la page 2.
    """
    # Spacer minuscule pour que la page 1 ne soit pas considérée vide
    story.append(Spacer(1, 1))
    story.append(PageBreak())


def _section_resume_executif(story, st, glob, annual, anomalies_data, nlp_data,
                              filters=None):
    """
    Résumé exécutif décisionnel.

    AUDIT v5.6 — Bug #1 + Bug #3 :
      - Gère croissance_globale = None (première année 2015 ou pas de N-1)
      - N'écrit plus "L'année X constitue le meilleur exercice" si on
        filtre par X (trivial). Idem pour la région.
    """
    story.append(Paragraph("Résumé Exécutif", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=C.BLEU, spaceAfter=10))
    story.append(Spacer(1, 8))

    filters = filters or {}
    f_year = filters.get("year")
    f_region = filters.get("region")
    f_category = filters.get("category")

    if glob:
        ca = _safe_float(glob.get("ca_total"), 0)
        crois_raw = glob.get("croissance_globale")
        best_yr = glob.get("meilleure_annee", "N/A")
        best_reg = glob.get("meilleure_region", "N/A")
        periode = glob.get("periode", "N/A")

        # ── Phrase 1 : CA + croissance (avec gestion None) ──
        if crois_raw is None:
            phrase1 = (
                f"Sur la période <b>{periode}</b>, le chiffre d'affaires total atteint "
                f"<b>{_fmt_money(ca)}</b>. La croissance YoY n'est pas calculable "
                f"(pas d'année comparable de référence)."
            )
        else:
            crois = _safe_float(crois_raw, 0)
            phrase1 = (
                f"Sur la période <b>{periode}</b>, le chiffre d'affaires total atteint "
                f"<b>{_fmt_money(ca)}</b> avec une croissance globale de "
                f"<b>{crois:+.1f}%</b>."
            )
        story.append(Paragraph(phrase1, st["body"]))
        story.append(Spacer(1, 4))

        # ── Phrase 2 : meilleure_annee / meilleure_region (Bug #3) ──
        # On évite les phrases triviales quand on filtre déjà par cette dimension
        elements_phrase2 = []
        if not f_year and best_yr and best_yr != "N/A":
            elements_phrase2.append(
                f"L'année <b>{best_yr}</b> constitue le meilleur exercice"
            )
        if not f_region and best_reg and best_reg != "N/A":
            elements_phrase2.append(
                f"la région <b>{best_reg}</b> domine en volume de ventes"
            )
        if elements_phrase2:
            phrase2 = ", ".join(elements_phrase2) + "."
            phrase2 = phrase2[0].upper() + phrase2[1:]  # capitalize début
            story.append(Paragraph(phrase2, st["body"]))
            story.append(Spacer(1, 6))

    # Bullets du rapport NLP (nettoyés)
    if nlp_data:
        for key in ["resume_bullet1", "resume_bullet2", "resume_bullet3"]:
            b = nlp_data.get(key)
            if b:
                story.append(Paragraph(f"• {_strip_markdown(b)}", st["bullet"]))
        story.append(Spacer(1, 6))

    alertes = [a for a in anomalies_data if a.get("niveau") == "ALERTE"]
    if alertes:
        story.append(Paragraph(f"<b>{len(alertes)} alerte(s) critique(s) détectée(s)</b> nécessitant l'attention du management.", st["body"]))

    story.append(Spacer(1, 10))


def _section_kpis(story, st, glob):
    """Dashboard KPIs — style éditorial broadsheet."""
    story.append(Paragraph("Indicateurs Clés de Performance", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C.BLEU,
                            spaceAfter=12))

    if not glob:
        story.append(Paragraph("Données indisponibles.", st["body"]))
        return

    crois_raw = glob.get("croissance_globale")
    crois_str = _fmt_growth(crois_raw, default_text="N/A")

    # 6 KPIs avec couleur d'accent personnalisée
    kpis = [
        (_fmt_money(_safe_float(glob.get("ca_total"), 0)), "CA TOTAL", C.BLEU),
        (crois_str, "CROISSANCE", C.VERT if (isinstance(crois_raw, (int, float)) and crois_raw >= 0) else C.ROUGE),
        (f"{_safe_int(glob.get('nb_commandes'), 0):,}", "COMMANDES", C.BLEU_F),
        (f"${_safe_float(glob.get('panier_moyen'), 0):,.0f}", "PANIER MOYEN", C.BLEU),
        (str(glob.get("meilleure_region", "N/A")), "TOP RÉGION", C.BLEU_F),
        (f"{_safe_int(glob.get('nb_clients'), 0):,}", "CLIENTS", C.BLEU),
    ]

    # Construit deux rangées de 3 cartes "broadsheet"
    def _make_kpi_card(value, label, accent):
        # Une cellule = une mini-table à 2 lignes (filet d'accent + valeur + label)
        val_para = Paragraph(value, st["kpi_val"])
        lbl_para = Paragraph(label, st["kpi_lbl"])
        inner = Table(
            [[val_para], [lbl_para]],
            colWidths=[5.3*cm],
        )
        inner.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (0, 0), 18),
            ("BOTTOMPADDING", (0, 0), (0, 0), 6),
            ("TOPPADDING", (0, 1), (0, 1), 0),
            ("BOTTOMPADDING", (0, 1), (0, 1), 14),
            ("LINEABOVE", (0, 0), (-1, 0), 3, accent),
            ("BACKGROUND", (0, 0), (-1, -1), C.BLANC),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, C.RULE),
            ("LINEBEFORE", (0, 0), (0, -1), 0.5, C.RULE_LIGHT),
            ("LINEAFTER", (-1, 0), (-1, -1), 0.5, C.RULE_LIGHT),
        ]))
        return inner

    # Rangée 1 et rangée 2
    for row_kpis in (kpis[:3], kpis[3:]):
        cards = [_make_kpi_card(v, l, a) for (v, l, a) in row_kpis]
        outer = Table([cards], colWidths=[5.6*cm] * 3)
        outer.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(outer)
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 6))


def _section_graphiques(story, st, annual, regions, categories, quarterly, monthly,
                        products, segments, region_detail,
                        sub_categories=None, filters=None):
    """Insère les graphiques avec sélection intelligente selon les filtres actifs."""
    filters = filters or {}
    f_year = filters.get("year")
    f_region = filters.get("region")
    f_category = filters.get("category")

    story.append(Paragraph("Analyses Graphiques", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=C.BLEU, spaceAfter=10))
    story.append(Spacer(1, 8))

    if f_year or f_region or f_category:
        active = []
        if f_year: active.append(f"Année: <b>{f_year}</b>")
        if f_region: active.append(f"Région: <b>{f_region}</b>")
        if f_category: active.append(f"Catégorie: <b>{f_category}</b>")
        story.append(Paragraph(
            f"<i>Filtres actifs — {' · '.join(active)}</i>", st["body_small"]
        ))
        story.append(Spacer(1, 6))

    charts = []

    # 1. CA par année OU décomposition trimestrielle
    if annual:
        if len(annual) == 1 and quarterly and len(quarterly) > 1:
            year_str = annual[0]['annee']
            ca_y = float(annual[0]['ca_annuel'])
            q_max = max(quarterly, key=lambda r: float(r['ventes_totales']))
            q_min = min(quarterly, key=lambda r: float(r['ventes_totales']))
            ca_txt = (
                f"Sur {year_str}, le CA total est de {_fmt_money(ca_y)}. "
                f"Le T{q_max['trimestre']} concentre le pic d'activité avec "
                f"{_fmt_money(float(q_max['ventes_totales']))}, tandis que le "
                f"T{q_min['trimestre']} affiche le creux ({_fmt_money(float(q_min['ventes_totales']))})."
            )
            fig = _chart_quarterly_breakdown(quarterly, year=year_str)
            if fig:
                charts.append(("Décomposition Trimestrielle", fig, ca_txt))
        else:
            ca_txt = (
                f"Le CA évolue de {_fmt_money(float(annual[0]['ca_annuel']))} en "
                f"{annual[0]['annee']} à {_fmt_money(float(annual[-1]['ca_annuel']))} en "
                f"{annual[-1]['annee']}, soit une progression de "
                f"{((float(annual[-1]['ca_annuel'])/float(annual[0]['ca_annuel']))-1)*100:.1f}% "
                f"sur la période."
            )
            charts.append(("Chiffre d'Affaires par Année", _chart_ca_annuel(annual), ca_txt))

    # 2. Croissance YoY
    if annual and len(annual) > 1:
        yoy_data = [(r["annee"], float(r["croissance_yoy"])) for r in annual
                    if r.get("croissance_yoy") is not None]
        if yoy_data:
            fig = _chart_croissance_yoy(annual)
            if fig:
                best = max(yoy_data, key=lambda x: x[1])
                croiss_txt = (
                    f"La meilleure performance annuelle est enregistrée en {best[0]} "
                    f"avec une croissance de {best[1]:+.1f}%. Les variations révèlent "
                    f"une dynamique irrégulière nécessitant une stabilisation."
                )
                charts.append(("Croissance Annuelle", fig, croiss_txt))

    # 3. Distribution régionale
    if regions and len(regions) > 1:
        reg_txt = (
            f"La région {regions[0]['region']} domine avec "
            f"{_fmt_money(float(regions[0]['ventes_totales']))}. L'écart avec la dernière "
            f"région ({regions[-1]['region']}) est de "
            f"{_fmt_money(float(regions[0]['ventes_totales']) - float(regions[-1]['ventes_totales']))}."
        )
        charts.append(("Distribution Régionale", _chart_regions(regions), reg_txt))

    # 4. Catégories
    if categories and len(categories) > 1:
        cat_top = categories[0]
        cat_txt = (
            f"La catégorie {cat_top['category']} représente la plus grande part du CA avec "
            f"{_fmt_money(float(cat_top['ventes_totales']))}. La diversification reste un "
            f"levier de croissance."
        )
        charts.append(("Répartition par Catégorie", _chart_categories(categories), cat_txt))

    # 5. Variations trimestrielles
    if quarterly and len(quarterly) > 1:
        # AUDIT v5.6 — gérer variation_pct None proprement
        # Bug #12 — légende plus précise : on distingue "chutes >20%" et
        # "chutes modérées 0-20%" pour ne plus afficher "0 chutes" alors que
        # le graphique en montre clairement.
        all_vars = [float(r["variation_pct"]) for r in quarterly
                    if r.get("variation_pct") is not None]
        neg_severe = [v for v in all_vars if v < -20]
        neg_moderee = [v for v in all_vars if -20 <= v < -2]
        if neg_severe:
            commentaire = (
                f"{len(neg_severe)} trimestre(s) présentent une chute supérieure à 20% — "
                f"investigation requise. {len(neg_moderee)} chute(s) modérée(s) entre -2% et -20%."
            )
        elif neg_moderee:
            commentaire = (
                f"Aucune chute sévère (>20%), mais {len(neg_moderee)} trimestre(s) en "
                f"recul modéré (-2% à -20%). Pattern de saisonnalité à surveiller."
            )
        else:
            commentaire = (
                "Toutes les variations QoQ restent positives ou stables sur le périmètre. "
                "Trajectoire sans creux significatif."
            )
        charts.append((
            "Variations Trimestrielles (QoQ)",
            _chart_quarterly(quarterly),
            commentaire
        ))

    # 6. CA trimestriel absolu
    if quarterly and len(quarterly) > 1:
        charts.append((
            "CA Trimestriel (Valeur Absolue)",
            _chart_ca_quarterly_abs(quarterly),
            "La tendance de fond est haussière malgré la volatilité. Les pics de fin "
            "d'année confirment un effet saisonnier favorable au T4."
        ))

    # 7. Évolution mensuelle
    if monthly and len(monthly) > 1:
        charts.append((
            "Évolution Mensuelle",
            _chart_monthly(monthly),
            "La courbe mensuelle révèle des cycles saisonniers avec des creux en début "
            "d'année et des pics en fin d'année. La moyenne mobile lisse ces variations."
        ))

    # 8. Top sous-catégories
    sub_to_use = sub_categories or products
    if sub_to_use:
        ctx_parts = []
        if f_year: ctx_parts.append(str(f_year))
        if f_region: ctx_parts.append(f_region)
        if f_category: ctx_parts.append(f_category)
        ctx = " · ".join(ctx_parts)
        if len(sub_to_use) >= 2:
            sub_txt = (
                f"{sub_to_use[0]['sub_category']} et {sub_to_use[1]['sub_category']} "
                f"concentrent une part majeure du CA. Une dépendance excessive à ces "
                f"sous-catégories constitue un risque."
            )
        elif len(sub_to_use) == 1:
            sub_txt = (
                f"{sub_to_use[0]['sub_category']} est la seule sous-catégorie active "
                f"sur ce périmètre, avec {_fmt_money(float(sub_to_use[0]['ventes_totales']))}."
            )
        else:
            sub_txt = "Analyse des sous-catégories."
        fig = _chart_sub_categories(sub_to_use, context=ctx)
        if fig:
            charts.append(("Top Sous-catégories", fig, sub_txt))

    # 9. Segments clients
    if segments and len(segments) >= 1:
        ctx_parts = []
        if f_year: ctx_parts.append(str(f_year))
        if f_region: ctx_parts.append(f_region)
        if f_category: ctx_parts.append(f_category)
        ctx = " · ".join(ctx_parts)
        seg_top = segments[0]
        seg_txt = f"Le segment {seg_top['segment']} domine avec {_fmt_money(float(seg_top['ventes_totales']))}."
        if len(segments) > 1:
            seg_txt += " Le développement des segments secondaires est un levier de diversification."
        fig = _chart_segments_breakdown(segments, context=ctx) or _chart_segments(segments)
        if fig:
            charts.append(("Mix Segments Clients", fig, seg_txt))

    # 10. Régions × années
    if region_detail and not f_region:
        nb_regions = len(set(r["region"] for r in region_detail))
        nb_years = len(set(r["annee"] for r in region_detail))
        if nb_regions > 1 and nb_years > 1:
            fig = _chart_region_year(region_detail)
            if fig:
                charts.append((
                    "Performance Régionale par Année", fig,
                    "L'analyse croisée région/année met en évidence les zones de croissance "
                    "et les marchés en stagnation."
                ))

    # 11. Volume de commandes
    if annual:
        if len(annual) == 1:
            if quarterly and len(quarterly) > 1:
                pass
            else:
                cmd_txt = (
                    f"L'année {annual[0]['annee']} totalise "
                    f"{int(annual[0]['nb_commandes']):,} commandes uniques "
                    f"({int(annual[0].get('nb_articles') or 0):,} articles)."
                )
                charts.append(("Volume de Commandes", _chart_commandes_annuelles(annual), cmd_txt))
        else:
            cmd_txt = (
                f"Le nombre de commandes uniques progresse de {int(annual[0]['nb_commandes']):,} "
                f"en {annual[0]['annee']} à {int(annual[-1]['nb_commandes']):,} en "
                f"{annual[-1]['annee']}, confirmant une expansion de la base d'activité."
            )
            charts.append(("Volume de Commandes (uniques)", _chart_commandes_annuelles(annual), cmd_txt))

    # 12. Panier moyen
    if annual and len(annual) > 1:
        baskets = [float(r["panier_moyen"]) for r in annual]
        delta = (baskets[-1] - baskets[0]) / baskets[0] * 100 if baskets[0] else 0
        panier_txt = (
            f"Le panier moyen évolue de ${baskets[0]:.0f} à ${baskets[-1]:.0f} "
            f"({delta:+.1f}%). Une stratégie d'upselling pourrait amplifier cette tendance."
        )
        charts.append(("Panier Moyen", _chart_panier_moyen(annual), panier_txt))

    # 13. Croissance cumulée
    if quarterly and len(quarterly) > 1:
        charts.append((
            "Croissance Cumulée", _chart_croissance_cumulee(quarterly),
            "La trajectoire de croissance cumulée confirme une tendance positive de long "
            "terme malgré des corrections ponctuelles."
        ))

    if not charts:
        story.append(Paragraph(
            "<i>Aucun graphique pertinent à afficher pour ce filtre. "
            "Élargissez le périmètre pour obtenir une analyse graphique.</i>",
            st["body"]
        ))
        return

    for i, (title, fig, interpretation) in enumerate(charts):
        if i > 0 and i % 2 == 0:
            story.append(PageBreak())
        story.append(Paragraph(title, st["h2"]))
        story.append(_fig_to_image(fig))
        _chart_interpretation(story, st, interpretation)
        story.append(Spacer(1, 12))


def _section_anomalies(story, st, anomalies_data):
    """Section alertes et anomalies — design éditorial.

    v5.8 — Titre adaptatif :
      - rien        → "Anomalies et Signaux"
      - alertes seulement → "Anomalies et Alertes"
      - infos seulement   → "Signaux Positifs"
      - mix         → "Anomalies, Alertes et Signaux Positifs"
    """
    if not anomalies_data:
        story.append(Paragraph("Anomalies et Signaux", st["h1"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=C.ROUGE,
                                spaceAfter=8))
        story.append(Paragraph(
            "Aucune anomalie ni signal positif détecté sur ce périmètre "
            "(seuil YoY : ±20%).",
            st["body"]))
        return

    alertes = [a for a in anomalies_data if a.get("niveau") == "ALERTE"]
    infos = [a for a in anomalies_data if a.get("niveau") != "ALERTE"]

    # Titre adaptatif selon le contenu réel
    if alertes and infos:
        titre = "Anomalies, Alertes et Signaux Positifs"
        couleur_filet = C.ROUGE
    elif alertes:
        titre = "Anomalies et Alertes"
        couleur_filet = C.ROUGE
    else:
        titre = "Signaux Positifs"
        couleur_filet = C.VERT

    story.append(Paragraph(titre, st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=couleur_filet,
                            spaceAfter=8))

    # ── Alertes critiques ──
    if alertes:
        story.append(Paragraph(
            f"<b>{len(alertes)}</b> alerte(s) critique(s) — chute YoY supérieure à 20%",
            st["h2"]
        ))
        story.append(Spacer(1, 4))

        for a in alertes:
            desc = a.get("description", "")
            try:
                var = abs(float(a.get("variation", 0)))
            except (TypeError, ValueError):
                var = 0
            impact = (
                f"Impact potentiel : baisse de {var:.1f}% du CA trimestriel. "
                f"Investigation recommandée (saisonnalité, marché, opérationnel)."
            )

            # Carte alerte : filet rouge à gauche + texte sur fond très léger
            desc_para = Paragraph(f"<b>{desc}</b>", st["alerte"])
            impact_para = Paragraph(impact, st["body_small"])
            card = Table(
                [[desc_para], [impact_para]],
                colWidths=[16*cm],
            )
            card.setStyle(TableStyle([
                ("LINEBEFORE", (0, 0), (0, -1), 3, C.ROUGE),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (0, 0), 8),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
                ("TOPPADDING", (0, -1), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ]))
            story.append(card)
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 8))

    # ── Signaux positifs ──
    if infos:
        story.append(Paragraph(
            f"<b>{len(infos)}</b> signal(aux) positif(s) — hausse YoY supérieure à 20%",
            st["h2"]
        ))
        story.append(Spacer(1, 4))

        for a in infos[:5]:
            desc = a.get("description", "")
            desc_para = Paragraph(desc, st["body"])
            card = Table(
                [[desc_para]],
                colWidths=[16*cm],
            )
            card.setStyle(TableStyle([
                ("LINEBEFORE", (0, 0), (0, -1), 3, C.VERT),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(card)
            story.append(Spacer(1, 4))

        story.append(Spacer(1, 8))


def _section_recommandations(story, st, glob, regions, categories, anomalies_data,
                              nlp_data=None, langue="fr"):
    """
    CHANTIER 4 — Recommandations hiérarchisées.
    
    Si la structure JSON (Chantier 1) est disponible dans nlp_data["structure"],
    on affiche les recommandations du LLM avec leurs badges
    PRIORITÉ / NIVEAU / CONFIANCE + justification chiffrée.
    
    Sinon, fallback sur des recommandations génériques (logique historique).
    """
    story.append(Paragraph(
        "Recommandations Stratégiques" if langue == "fr" else "Strategic Recommendations",
        st["h1"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=C.VERT, spaceAfter=10))
    story.append(Spacer(1, 8))

    # ────────────────────────────────────────────────────────────
    # Cas 1 : structure JSON disponible → recos hiérarchisées LLM
    # ────────────────────────────────────────────────────────────
    structure = None
    if isinstance(nlp_data, dict):
        structure = nlp_data.get("structure")
    
    recos_structurees = []
    if isinstance(structure, dict):
        recos_structurees = structure.get("recommendations") or []
    
    if recos_structurees:
        _rendre_recos_hierarchisees(story, st, recos_structurees, langue=langue)
        return

    # ────────────────────────────────────────────────────────────
    # Cas 2 : Fallback historique (aucune structure disponible)
    # ────────────────────────────────────────────────────────────
    recos = []

    alertes = [a for a in anomalies_data if a.get("niveau") == "ALERTE"]
    if alertes:
        recos.append(
            "Mettre en place un dispositif de surveillance renforcée sur les T1 "
            "qui présentent un pattern récurrent de baisse significative."
            if langue == "fr" else
            "Set up reinforced monitoring on Q1 which shows a recurring "
            "significant drop pattern."
        )

    if regions and len(regions) >= 2:
        top = regions[0]
        bottom = regions[-1]
        if langue == "fr":
            recos.append(
                f"Renforcer les investissements commerciaux sur la région {top['region']} "
                f"qui génère le meilleur rendement, tout en développant un plan de relance "
                f"pour la région {bottom['region']}."
            )
        else:
            recos.append(
                f"Reinforce commercial investment on {top['region']} which delivers "
                f"the best returns, while building a recovery plan for {bottom['region']}."
            )

    if categories and len(categories) >= 2:
        if float(categories[0]["ventes_totales"]) > float(categories[-1]["ventes_totales"]) * 1.3:
            if langue == "fr":
                recos.append(
                    f"Diversifier le portefeuille produits pour réduire la dépendance "
                    f"à la catégorie {categories[0]['category']} et équilibrer le mix de revenus."
                )
            else:
                recos.append(
                    f"Diversify product portfolio to reduce dependence on "
                    f"{categories[0]['category']} and balance the revenue mix."
                )

    if glob:
        crois = _safe_float(glob.get("croissance_globale"), 0)
        if crois > 30:
            recos.append(
                "Capitaliser sur la dynamique de forte croissance en augmentant "
                "la capacité opérationnelle et en anticipant les besoins logistiques."
                if langue == "fr" else
                "Capitalize on strong growth momentum by scaling operational "
                "capacity and anticipating logistics needs."
            )
        elif crois < 5:
            recos.append(
                "Stimuler la croissance par des initiatives commerciales innovantes "
                "(promotions ciblées, fidélisation, expansion géographique)."
                if langue == "fr" else
                "Stimulate growth through innovative commercial initiatives "
                "(targeted promotions, loyalty, geographical expansion)."
            )

    recos.append(
        "Automatiser le suivi des KPIs via le dashboard et programmer des "
        "revues trimestrielles pour piloter la performance en temps réel."
        if langue == "fr" else
        "Automate KPI tracking via the dashboard and schedule quarterly "
        "reviews to steer performance in real time."
    )
    recos.append(
        "Investiguer les causes structurelles de la volatilité trimestrielle "
        "observée et définir des actions correctrices préventives."
        if langue == "fr" else
        "Investigate structural causes of observed quarterly volatility and "
        "define preventive corrective actions."
    )

    for i, r in enumerate(recos, 1):
        story.append(Paragraph(f"<b>{i}.</b> {r}", st["reco"]))
    story.append(Spacer(1, 10))


# ═══════════════════════════════════════════════════════════════════════════
#  CHANTIER 4 — HELPER : RECOS HIÉRARCHISÉES AVEC BADGES
# ═══════════════════════════════════════════════════════════════════════════

# Mapping des couleurs pour les badges
_COULEURS_PRIO = {
    "haute":   "#E24B4A",   # rouge
    "moyenne": "#EF9F27",   # orange
    "basse":   "#639922",   # vert
}
_COULEURS_NIV = {
    "strategique":  "#0D3F6E",   # bleu foncé
    "tactique":     "#1A5FA8",   # bleu
    "operationnel": "#5A8BB8",   # bleu clair
}
_COULEURS_CONF = {
    "elevee":  "#639922",   # vert
    "moyenne": "#888780",   # gris
    "faible":  "#E24B4A",   # rouge
}

_LIBELLES_PRIO_FR = {"haute": "HAUTE", "moyenne": "MOYENNE", "basse": "BASSE"}
_LIBELLES_PRIO_EN = {"haute": "HIGH", "moyenne": "MEDIUM", "basse": "LOW"}

_LIBELLES_NIV_FR = {
    "strategique": "Stratégique",
    "tactique": "Tactique",
    "operationnel": "Opérationnel",
}
_LIBELLES_NIV_EN = {
    "strategique": "Strategic",
    "tactique": "Tactical",
    "operationnel": "Operational",
}

_LIBELLES_CONF_FR = {"elevee": "Élevée", "moyenne": "Moyenne", "faible": "Faible"}
_LIBELLES_CONF_EN = {"elevee": "High",    "moyenne": "Medium",  "faible": "Low"}


def _badge(texte: str, couleur_hex: str) -> str:
    """Construit un badge HTML coloré pour Paragraph ReportLab."""
    return (
        f'<font color="white" backColor="{couleur_hex}">'
        f'&nbsp;<b>{texte}</b>&nbsp;'
        f'</font>'
    )


def _rendre_recos_hierarchisees(story, st, recos: list, langue: str = "fr"):
    """
    Rend les recommandations hiérarchisées (Chantier 4) avec :
      - badges [PRIORITÉ] [NIVEAU] [CONFIANCE]
      - action en corps de texte
      - justification chiffrée en italique
      - synthèse comptable en fin (combien de recos par priorité)
    """
    lib_prio = _LIBELLES_PRIO_FR if langue == "fr" else _LIBELLES_PRIO_EN
    lib_niv  = _LIBELLES_NIV_FR  if langue == "fr" else _LIBELLES_NIV_EN
    lib_conf = _LIBELLES_CONF_FR if langue == "fr" else _LIBELLES_CONF_EN

    # Intro contextuelle (explique la logique des badges)
    intro = (
        "Les recommandations ci-dessous sont hiérarchisées par <b>priorité</b>, "
        "<b>niveau d'impact</b> (stratégique / tactique / opérationnel) et "
        "<b>niveau de confiance</b> (lié à la stabilité des données)."
        if langue == "fr" else
        "Recommendations below are ranked by <b>priority</b>, "
        "<b>impact level</b> (strategic / tactical / operational) and "
        "<b>confidence level</b> (based on data stability)."
    )
    story.append(Paragraph(intro, st["body_small"]))
    story.append(Spacer(1, 6))

    # Libellés pour justification
    lbl_just = "Justification" if langue == "fr" else "Justification"

    for i, r in enumerate(recos, 1):
        prio = (r.get("priorite") or "moyenne").lower()
        niv  = (r.get("niveau")   or "tactique").lower()
        conf = (r.get("confiance") or "moyenne").lower()

        badge_prio = _badge(
            f"PRIO {lib_prio.get(prio, prio.upper())}",
            _COULEURS_PRIO.get(prio, "#888780")
        )
        badge_niv = _badge(
            lib_niv.get(niv, niv.capitalize()),
            _COULEURS_NIV.get(niv, "#1A5FA8")
        )
        badge_conf = _badge(
            f"{('Confiance' if langue == 'fr' else 'Conf.')} {lib_conf.get(conf, conf)}",
            _COULEURS_CONF.get(conf, "#888780")
        )

        # Ligne de badges
        story.append(Paragraph(
            f"<b>{i}.</b> {badge_prio} &nbsp; {badge_niv} &nbsp; {badge_conf}",
            st["body"]
        ))
        # Action
        story.append(Paragraph(
            f"<b>→</b> {_strip_markdown(r.get('action', ''))}",
            st["reco"]
        ))
        # Justification (si fournie)
        just = r.get("justification")
        if just:
            story.append(Paragraph(
                f"<i>{lbl_just} : {_strip_markdown(just)}</i>",
                st["body_small"]
            ))
        story.append(Spacer(1, 8))

    # ── Synthèse comptable ──
    nb_haute = sum(1 for r in recos if (r.get("priorite") or "").lower() == "haute")
    nb_moyenne = sum(1 for r in recos if (r.get("priorite") or "").lower() == "moyenne")
    nb_basse = sum(1 for r in recos if (r.get("priorite") or "").lower() == "basse")

    synthese = (
        f"<b>Synthèse :</b> {len(recos)} recommandation(s) — "
        f"{nb_haute} priorité haute, {nb_moyenne} moyenne, {nb_basse} basse."
        if langue == "fr" else
        f"<b>Summary:</b> {len(recos)} recommendation(s) — "
        f"{nb_haute} high priority, {nb_moyenne} medium, {nb_basse} low."
    )
    story.append(Paragraph(synthese, st["body_small"]))
    story.append(Spacer(1, 10))


# ═══════════════════════════════════════════════════════════════════════════
#  v5.3 — SECTION NLP : texte paragraphé sans markdown
# ═══════════════════════════════════════════════════════════════════════════

def _section_nlp(story, st, nlp_data):
    """Rapport NLP généré automatiquement - texte naturel paragraphé."""
    story.append(Paragraph("Rapport NLP Automatique", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=C.BLEU, spaceAfter=10))
    story.append(Spacer(1, 8))

    if not nlp_data or not nlp_data.get("rapport_complet"):
        story.append(Paragraph(
            "Aucun rapport NLP disponible pour les filtres sélectionnés.",
            st["body"]
        ))
        return

    # En-tête avec contexte
    periode = nlp_data.get("periode", "")
    filters_str = []
    if nlp_data.get("filter_year"):
        filters_str.append(f"Année {nlp_data['filter_year']}")
    if nlp_data.get("filter_region"):
        filters_str.append(f"Région {nlp_data['filter_region']}")
    if nlp_data.get("filter_category"):
        filters_str.append(f"Catégorie {nlp_data['filter_category']}")
    filtre_str = " · ".join(filters_str) if filters_str else "Périmètre global"

    story.append(Paragraph(
        f"<b>Analyse générée dynamiquement</b> — {filtre_str}"
        f"{f' ({periode})' if periode else ''}",
        st["body"]
    ))
    story.append(Spacer(1, 6))

    # Rapport principal — nettoyé et découpé en paragraphes
    rapport = _strip_markdown(nlp_data["rapport_complet"])

    for paragraphe in rapport.split("\n\n"):
        ligne = paragraphe.strip().replace("\n", " ")
        if ligne:
            story.append(Paragraph(ligne, st["body"]))
            story.append(Spacer(1, 6))

    # Bullets résumé
    bullets = []
    for k in ("resume_bullet1", "resume_bullet2", "resume_bullet3"):
        if nlp_data.get(k):
            bullets.append(_strip_markdown(nlp_data[k]))

    if bullets:
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Points clés :</b>", st["body"]))
        for b in bullets:
            story.append(Paragraph(f"• {b}", st["body"]))

    # Score
    score = nlp_data.get("score")
    mention = nlp_data.get("mention", "")
    if score is not None:
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"<b>Score de performance :</b> {score}/100 — {mention}",
            st["body"]
        ))
    story.append(Spacer(1, 10))


# ═══════════════════════════════════════════════════════════════════════════
#  CHANTIER 3 — SECTION "QUALITÉ NLP DU RAPPORT"
# ═══════════════════════════════════════════════════════════════════════════

def _section_qualite_nlp(story, st, nlp_data: dict, faits_pg: dict = None):
    """
    Section dédiée au score qualité NLP (Chantier 3).
    Affiche le score /100, la mention, le détail des 6 critères
    et les suggestions d'amélioration.
    Insérée juste après la section Rapport NLP.

    FIX BUG #2 : si on doit recalculer le score à la volée, on passe
    désormais les vrais faits au lieu de {} → le critère "couverture des
    faits" peut enfin scorer correctement (avant : toujours 0).
    """
    # Récupère le score NLP depuis nlp_data (en cache ou généré dynamiquement)
    score_nlp = nlp_data.get("score_nlp") if nlp_data else None
    mention_nlp = nlp_data.get("score_nlp_mention", "") if nlp_data else ""
    details_nlp = nlp_data.get("score_nlp_details", {}) if nlp_data else {}
    lacunes_nlp = nlp_data.get("score_nlp_lacunes", []) if nlp_data else []

    # Si le score NLP n'est pas en cache, essaye de le calculer à la volée
    if score_nlp is None and nlp_data and nlp_data.get("rapport_complet"):
        try:
            from nlp_quality_score import evaluer_qualite_rapport
            rapport_txt = _strip_markdown(nlp_data["rapport_complet"])
            # Important : on passe les vrais faits (pas {}), sinon le
            # critère 1 "couverture des faits" est toujours à 0.
            res = evaluer_qualite_rapport(
                rapport_texte=rapport_txt,
                faits=faits_pg or {},
            )
            score_nlp     = res.get("score_nlp")
            mention_nlp   = res.get("mention", "")
            details_nlp   = res.get("details", {})
            lacunes_nlp   = res.get("lacunes", [])
        except Exception:
            pass

    if score_nlp is None:
        return  # Rien à afficher

    story.append(Paragraph("Qualité du Rapport NLP", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=C.BLEU, spaceAfter=10))
    story.append(Spacer(1, 8))

    # Score global
    if score_nlp >= 85:
        couleur_score = C.VERT
    elif score_nlp >= 55:
        couleur_score = C.ORANGE
    else:
        couleur_score = C.ROUGE

    story.append(Paragraph(
        f"<b>Score global :</b> <font color='{couleur_score.hexval()}'>"
        f"{score_nlp}/100 — {mention_nlp}</font>",
        st["body"]
    ))
    story.append(Spacer(1, 8))

    # Tableau détail des critères
    LIBELLES = {
        "couverture_faits":         "Couverture des faits clés",
        "ancrage_numerique":        "Ancrage numérique",
        "presence_recommandations": "Présence de recommandations",
        "clarte_lisibilite":        "Clarté / lisibilité",
        "absence_repetition":       "Absence de répétition",
        "ton_business":             "Ton business adapté",
    }

    if details_nlp:
        table_data = [["Critère", "Points", "Max", "Barre"]]
        for cle, d in details_nlp.items():
            # FIX BUG #2 : nlp_quality_score expose "score" et "max",
            # PAS "points". On lit "score" en priorité, "points" en fallback
            # pour compat ascendante. Avant : tout affichait 0.
            pts = int(d.get("score", d.get("points", 0)) or 0)
            maxi = int(d.get("max", 0) or 0)
            # Sécurité : limiter pour ne pas générer de barres absurdes
            pts = max(0, min(pts, maxi)) if maxi else 0
            barre = "█" * pts + "░" * max(0, maxi - pts)
            table_data.append([
                LIBELLES.get(cle, cle),
                str(pts),
                str(maxi),
                barre,
            ])

        from reportlab.platypus import Table as RLTable, TableStyle as RLTableStyle
        from reportlab.lib import colors as rl_colors

        tbl = RLTable(table_data, colWidths=[7*cm, 1.5*cm, 1.5*cm, 5*cm])
        tbl.setStyle(RLTableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), C.BLEU_F),
            ("TEXTCOLOR",    (0, 0), (-1, 0), C.BLANC),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("BACKGROUND",   (0, 1), (-1, -1), C.GRIS_CL),
            ("ALIGN",        (1, 0), (2, -1), "CENTER"),
            ("FONTNAME",     (3, 1), (3, -1), "Courier"),
            ("GRID",         (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C.BLANC, C.GRIS_CL]),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 8))

    # Suggestions d'amélioration
    if lacunes_nlp:
        story.append(Paragraph("<b>Suggestions d'amélioration :</b>", st["body"]))
        for lacune in lacunes_nlp[:5]:
            story.append(Paragraph(f"[!]{lacune}", st["body"]))

    story.append(Spacer(1, 10))

def _section_methodologie(story, st):
    """Section méthodologie."""
    story.append(PageBreak())
    story.append(Paragraph("Méthodologie", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=C.GRIS, spaceAfter=10))
    story.append(Spacer(1, 8))

    steps = [
        ("Source de données", "Dataset Superstore Sales (Kaggle) — 9 800 transactions commerciales couvrant la période 2015-2018, incluant ventes, régions, catégories et segments clients."),
        ("Traitement Spark", "Apache Spark est utilisé pour le chargement, le nettoyage et la transformation des données brutes. Les colonnes temporelles (année, trimestre, mois) sont dérivées automatiquement."),
        ("Calcul des KPIs", "12 indicateurs clés sont calculés par Spark : CA annuel, variations trimestrielles, performance régionale, catégorielle, segments, top produits, saisonnalité, etc."),
        ("Stockage PostgreSQL", "Tous les KPIs sont stockés dans PostgreSQL (11 tables) qui constitue la source unique de vérité du système."),
        ("Extraction NLTK (Chantier 2)", "NLTK effectue l'extraction linguistique (tokenisation, lemmatisation, NER, POS tagging) puis classifie chaque phrase du rapport en intentions business (tendance, anomalie, risque, recommandation) pour mesurer la couverture métier du texte généré."),
        ("Génération NLP 2-étapes (Chantier 1)", "Mistral API (architecture Transformer) est d'abord interrogé pour produire un JSON structuré (résumé, risques, opportunités, recommandations hiérarchisées) validé contre les faits ; ce JSON est ensuite transformé en texte narratif — ce qui garantit cohérence et anti-hallucination."),
        ("Tonalité business (lexique métier)", "Plutôt que d'utiliser le sentiment VADER, généraliste et peu adapté au B2B, le pipeline évalue une tonalité métier via un lexique dédié (croissance/décroissance, opportunités/risques, volatilité) et produit un score business borné à [-100, +100]."),
        ("Recommandations hiérarchisées (Chantier 4)", "Chaque recommandation porte trois attributs : priorité (haute/moyenne/basse), niveau (stratégique/tactique/opérationnel) et confiance (élevée/moyenne/faible) dérivée de la volatilité des données. Elles sont triées automatiquement par priorité."),
        ("Score qualité NLP (Chantier 3)", "Le rapport final est évalué sur 6 critères (couverture des faits, ancrage numérique, présence de recommandations, clarté, absence de répétition, ton business) pour produire un score /100 et des suggestions d'amélioration."),
        ("API FastAPI", "Une API REST sert de couche intermédiaire entre PostgreSQL et les consommateurs (dashboard, PDF, chatbot), avec des endpoints dédiés pour les KPIs, le rapport NLP, la structure hiérarchisée et le score qualité."),
        ("Génération du rapport", "Ce PDF est généré dynamiquement par ReportLab + Matplotlib à partir des données PostgreSQL via le module db.py. Aucune valeur n'est hardcodée."),
    ]

    for title, desc in steps:
        story.append(Paragraph(f"<b>{title}</b>", st["methodo"]))
        story.append(Paragraph(desc, st["methodo"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))


def _section_conclusion(story, st, glob, anomalies_data):
    """Conclusion du rapport."""
    story.append(Paragraph("Conclusion", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=C.BLEU, spaceAfter=10))
    story.append(Spacer(1, 8))

    ca = _safe_float(glob.get("ca_total"), 0) if glob else 0
    crois_raw = glob.get("croissance_globale") if glob else None
    nb_alertes = len([a for a in anomalies_data if a.get("niveau") == "ALERTE"])
    nb_infos = len([a for a in anomalies_data if a.get("niveau") != "ALERTE"])

    # v5.8 — Phrase de bilan adaptative selon le mix d'anomalies/signaux.
    # Avant : "Toutefois, 0 alerte(s) identifiée(s) appellent une vigilance"
    # → contradictoire et trompeur quand il y a en fait 5 signaux positifs.
    if nb_alertes > 0 and nb_infos > 0:
        bilan = (f"Le périmètre fait apparaître <b>{nb_alertes} alerte(s) critique(s)</b> "
                 f"appelant une vigilance particulière, ainsi que <b>{nb_infos} signal(aux) positif(s)</b> "
                 f"de hausse exceptionnelle à exploiter.")
    elif nb_alertes > 0:
        bilan = (f"<b>{nb_alertes} alerte(s) critique(s)</b> identifiée(s) appellent une vigilance "
                 f"particulière, notamment sur la volatilité trimestrielle et les patterns saisonniers.")
    elif nb_infos > 0:
        bilan = (f"Aucune alerte critique sur ce périmètre, mais <b>{nb_infos} signal(aux) positif(s)</b> "
                 f"de hausse exceptionnelle (YoY > 20%) sont à capitaliser pour la suite.")
    else:
        bilan = ("Aucune anomalie ni signal positif détecté sur ce périmètre "
                 "(seuil YoY : ±20%) — la trajectoire reste stable.")

    # AUDIT v5.6 Bug #1 — gestion croissance None (première année)
    if crois_raw is None:
        story.append(Paragraph(
            f"L'analyse complète du dataset Superstore révèle un chiffre d'affaires cumulé de "
            f"<b>{_fmt_money(ca)}</b> sur la période étudiée. La croissance YoY n'est pas "
            f"calculable sur ce périmètre (pas d'année comparable disponible). {bilan}",
            st["body"]))
    else:
        crois = _safe_float(crois_raw, 0)
        story.append(Paragraph(
            f"L'analyse complète du dataset Superstore révèle une entreprise en <b>croissance de {crois:+.1f}%</b> "
            f"sur la période étudiée, avec un chiffre d'affaires cumulé de <b>{_fmt_money(ca)}</b>. "
            f"{bilan}", st["body"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Ce rapport a été généré automatiquement par le système <b>NLP Business Reporting</b>, "
        "démontrant la capacité d'un pipeline analytique intégré (Spark + NLP + PostgreSQL) à produire "
        "des analyses décisionnelles fiables, reproductibles et personnalisables.", st["body"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "La valeur de ce système réside dans sa capacité à être exécuté régulièrement, "
        "avec des filtres dynamiques (année, région, catégorie), offrant aux décideurs "
        "un outil de pilotage opérationnel en continu.", st["body"]))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C.RULE_LIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} — "
        f"Pipeline NLP Reporting v5.6 (Spark · NLTK · Mistral · PostgreSQL · ChromaDB · FastAPI)", st["footer"]))


# ═══════════════════════════════════════════════════════════════════════════
#  v5.3 — CONSTRUCTION DES FAITS DEPUIS POSTGRESQL (pour génération dynamique)
# ═══════════════════════════════════════════════════════════════════════════

def _build_faits_from_pg(year=None, region=None, category=None):
    """Assemble les faits depuis PostgreSQL filtré pour alimenter nlp_transformers."""
    annual = get_kpi_annual_filtered(year, region, category) \
        if _has_filters(year, region, category) else get_kpi_annual()
    quarterly = get_kpi_quarterly(year, region, category)
    top = get_kpi_top_products(year, region, category)
    segments = get_kpi_segments_summary(year, region, category)
    summary = get_kpi_filtered_summary(year, region, category) or get_kpi_global()

    faits_annuel = []
    for r in (annual or []):
        try:
            faits_annuel.append({
                "annee": int(r.get("annee")),
                "ca": float(r.get("ca_annuel", 0) or 0),
                "commandes": int(r.get("nb_commandes", 0) or 0),
                "panier_moyen": float(r.get("panier_moyen", 0) or 0),
                "croissance_yoy": float(r["croissance_yoy"]) if r.get("croissance_yoy") is not None else None,
            })
        except Exception:
            continue
    faits_annuel.sort(key=lambda x: x["annee"])

    if not faits_annuel:
        ca = float(summary.get("ca_total", 0) or 0) if summary else 0
        nb = int(summary.get("nb_commandes", 0) or 0) if summary else 0
        faits_annuel = [{
            "annee": year or 2018, "ca": ca, "commandes": nb,
            "panier_moyen": (ca / nb) if nb else 0, "croissance_yoy": None,
        }]

    faits_var = []
    for r in (quarterly or []):
        try:
            # AUDIT v5.6 Bug #4 — préserver None
            var_raw = r.get("variation_pct")
            faits_var.append({
                "annee": int(r.get("annee")),
                "trimestre": int(r.get("trimestre")),
                "ventes": float(r.get("ventes_totales", 0) or 0),
                "variation": float(var_raw) if var_raw is not None else None,
            })
        except Exception:
            continue

    faits_top = [{
        "nom": r.get("sub_category", "N/A"),
        "ventes": float(r.get("ventes_totales", 0) or 0),
        "qte": int(r.get("quantite_vendue", 0) or 0),
    } for r in (top or [])[:3]]
    while len(faits_top) < 3:
        faits_top.append({"nom": "N/A", "ventes": 0.0, "qte": 0})

    faits_segs = {r.get("segment", "N/A"): {"ventes_total": float(r.get("ventes_totales", 0) or 0)}
                  for r in (segments or [])}

    best_year = max(faits_annuel, key=lambda x: x["ca"])

    # AUDIT v5.6 Bug #4 — ignorer variations None pour best/worst trim
    var_calculables = [v for v in faits_var if v.get("variation") is not None]
    if var_calculables:
        best_trim = max(var_calculables, key=lambda x: x["variation"])
        worst_trim = min(var_calculables, key=lambda x: x["variation"])
    elif faits_var:
        best_trim = max(faits_var, key=lambda x: x["ventes"])
        worst_trim = min(faits_var, key=lambda x: x["ventes"])
        best_trim = dict(best_trim, variation=None)
        worst_trim = dict(worst_trim, variation=None)
    else:
        best_trim = {"annee": best_year["annee"], "trimestre": 1,
                     "ventes": best_year["ca"], "variation": 0.0}
        worst_trim = best_trim

    ca_d = faits_annuel[0]["ca"] or 1
    ca_f = faits_annuel[-1]["ca"]

    # AUDIT v5.6 Bug #14 — quand on est sur une année unique sans N-1
    # comparable, on retourne None (pas de croissance "globale" calculable).
    # Avant : on faisait `(T4-T1)/T1 * 100` qui produisait des "+140%"
    # absurdes étiquetés "Croissance globale 2015". Inacceptable.
    if len(faits_annuel) == 1:
        yoy_sum = summary.get("croissance_globale") if summary else None
        if yoy_sum is not None:
            try:
                cg = round(float(yoy_sum), 1)
            except (TypeError, ValueError):
                cg = None
        else:
            # avant : on calculait (T_last - T_first)/T_first → faux
            # maintenant : on assume None et on laisse le formatteur
            # afficher "non calculable".
            cg = None
    else:
        cg = round((ca_f - ca_d) / ca_d * 100, 1) if ca_d else 0
    ad, af = faits_annuel[0]["annee"], faits_annuel[-1]["annee"]
    periode = str(ad) if ad == af else f"{ad}–{af}"

    var_abs = [abs(v["variation"]) for v in faits_var
               if v.get("variation") is not None]
    vol_val = round(sum(var_abs) / len(var_abs), 1) if var_abs else 0.0
    vol_niv = "Élevée" if vol_val > 30 else ("Modérée" if vol_val > 15 else "Faible")
    # Bug #14 — tendance uniquement si cg numérique
    if cg is None:
        tendance = "N/A"
    else:
        tendance = "Hausse" if cg > 2 else ("Baisse" if cg < -2 else "Stable")

    return {
        "periode": periode, "annuel": faits_annuel, "variations": faits_var,
        "top_produits": faits_top, "segments": faits_segs,
        "meilleure_annee": best_year, "pire_annee": min(faits_annuel, key=lambda x: x["ca"]),
        "meilleur_trim": best_trim, "pire_trim": worst_trim,
        "croissance_globale": cg,
        "tendance_globale": {"label": tendance},
        "volatilite": {"niveau": vol_niv, "valeur": vol_val},
        "region_focus": region, "categorie_focus": category, "regions": {},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  v5.3 — RÉCUPÉRATION NLP INTELLIGENTE (cache + génération dynamique)
# ═══════════════════════════════════════════════════════════════════════════

def _recuperer_nlp_data_pour_pdf(report_type, langue, year, region, category,
                                  resultats_externes: dict = None):
    """
    1. Priorise `resultats_externes` si fourni (main.py passe le resultats complet,
       avec structure + tonalité)
    2. Tente le cache pour les filtres exacts
    3. Si absent → GÉNÈRE à la volée depuis PostgreSQL filtré
    4. Nettoie le markdown
    
    Chantier 4 : retourne aussi la structure JSON (recommandations hiérarchisées)
    Finition  : retourne aussi la tonalité business et les thèmes
    """
    # ── 0. Cas prioritaire : main.py nous passe le résultats complet ──
    if isinstance(resultats_externes, dict) and resultats_externes.get("rapport"):
        rapport_clean = _strip_markdown(resultats_externes["rapport"])
        resume = resultats_externes.get("resume") or {}
        bullets = resume.get("bullets", []) if isinstance(resume, dict) else []
        score_data = resultats_externes.get("score") or {}
        score_nlp_obj = resultats_externes.get("score_nlp") or {}
        nltk_rapport = resultats_externes.get("nltk_rapport") or {}
        structure = resultats_externes.get("structure") or {}
        return {
            "rapport_complet": rapport_clean,
            "resume_bullet1": _strip_markdown(bullets[0]) if len(bullets) > 0 else "",
            "resume_bullet2": _strip_markdown(bullets[1]) if len(bullets) > 1 else "",
            "resume_bullet3": _strip_markdown(bullets[2]) if len(bullets) > 2 else "",
            "score": int(score_data.get("score", 0) or 0),
            "mention": score_data.get("mention", ""),
            "langue": langue, "report_type": report_type,
            "filter_year": year, "filter_region": region, "filter_category": category,
            "generated_at": datetime.utcnow().isoformat(),
            "source": "pipeline",
            # Chantier 4 — structure JSON avec recos hiérarchisées
            "structure": structure,
            # Chantier 3 — score qualité NLP
            "score_nlp": score_nlp_obj.get("score_nlp"),
            "score_nlp_mention": score_nlp_obj.get("mention"),
            "score_nlp_details": score_nlp_obj.get("details"),
            "score_nlp_lacunes": score_nlp_obj.get("lacunes"),
            # Finition — tonalité business + thèmes (Chantier 2 enrichi)
            "tonalite_business": nltk_rapport.get("tonalite_business"),
            "themes_business":   nltk_rapport.get("themes"),
            "couverture":        nltk_rapport.get("couverture"),
        }

    # ── 1. Cache ──
    nlp_data = get_nlp_report(
        report_type=report_type, lang=langue,
        year=year, region=region, category=category
    )

    if nlp_data and nlp_data.get("rapport_complet"):
        nlp_data = dict(nlp_data)
        nlp_data["rapport_complet"] = _strip_markdown(nlp_data["rapport_complet"])
        for k in ("resume_bullet1", "resume_bullet2", "resume_bullet3"):
            if nlp_data.get(k):
                nlp_data[k] = _strip_markdown(nlp_data[k])
        nlp_data["source"] = "cache"

        # AUDIT v5.6.2 Bug #14 résiduel — On REGÉNÈRE les bullets de
        # résumé exécutif depuis les faits récents, car le cache DB peut
        # contenir des bullets buggés ("+0.0%" pour 2015) si le main.py
        # a été exécuté avant la v5.6.1. Coût : ~1ms (pas d'appel Mistral).
        try:
            from src.nlp_transformers import generer_resume_executif
            faits_pg = _build_faits_from_pg(year=year, region=region, category=category)
            resume = generer_resume_executif(faits_pg, langue=langue)
            bullets_live = resume.get("bullets", []) if isinstance(resume, dict) else []
            if bullets_live:
                nlp_data["resume_bullet1"] = _strip_markdown(bullets_live[0]) \
                    if len(bullets_live) > 0 else nlp_data.get("resume_bullet1", "")
                nlp_data["resume_bullet2"] = _strip_markdown(bullets_live[1]) \
                    if len(bullets_live) > 1 else nlp_data.get("resume_bullet2", "")
                nlp_data["resume_bullet3"] = _strip_markdown(bullets_live[2]) \
                    if len(bullets_live) > 2 else nlp_data.get("resume_bullet3", "")
        except Exception as e:
            logger.warning(f"Régénération bullets depuis cache PDF échouée: {e}")

        # Le cache ne contient pas la structure → on la reconstruit à la volée
        # pour que le PDF puisse afficher des recos hiérarchisées
        try:
            from src.nlp_transformers import generer_structure_json
            faits_pg = _build_faits_from_pg(year=year, region=region, category=category)
            nlp_data["structure"] = generer_structure_json(faits_pg, langue=langue)
        except Exception as e:
            logger.warning(f"Structure JSON non reconstruite depuis cache: {e}")
            nlp_data["structure"] = {}
        # Pareil pour la tonalité business
        try:
            from src.nlp_nltk import evaluer_tonalite_business, extraire_themes_business
            nlp_data["tonalite_business"] = evaluer_tonalite_business(
                nlp_data["rapport_complet"], langue=langue
            )
            nlp_data["themes_business"] = extraire_themes_business(
                nlp_data["rapport_complet"], langue=langue, top_n=5
            )
        except Exception as e:
            logger.warning(f"Tonalité business non calculée: {e}")
        return nlp_data

    # ── 2. Génération dynamique ──
    logger.info(f"Pas de cache NLP — génération dynamique (year={year}, region={region}, category={category})")
    try:
        from src.nlp_transformers import (
            generer_rapport_avec_structure, generer_resume_executif, calculer_score
        )
        from src.nlp_nltk import evaluer_tonalite_business, extraire_themes_business

        faits = _build_faits_from_pg(year=year, region=region, category=category)
        # Chantier 4 : on récupère rapport + structure
        res = generer_rapport_avec_structure(faits, langue=langue)
        rapport = _strip_markdown(res["rapport"])
        structure = res.get("structure", {})

        try:
            resume = generer_resume_executif(faits, langue=langue)
            bullets = resume.get("bullets", []) if isinstance(resume, dict) else []
        except Exception as e:
            logger.warning(f"resume_executif échoué: {e}")
            bullets = []

        try:
            score_data = calculer_score(faits)
            score_val = int(score_data.get("score", 0))
            mention = score_data.get("mention", "")
        except Exception as e:
            logger.warning(f"calculer_score échoué: {e}")
            score_val, mention = 0, ""

        # Tonalité & thèmes
        tonalite = evaluer_tonalite_business(rapport, langue=langue)
        themes   = extraire_themes_business(rapport, langue=langue, top_n=5)

        return {
            "rapport_complet": rapport,
            "resume_bullet1": _strip_markdown(bullets[0]) if len(bullets) > 0 else "",
            "resume_bullet2": _strip_markdown(bullets[1]) if len(bullets) > 1 else "",
            "resume_bullet3": _strip_markdown(bullets[2]) if len(bullets) > 2 else "",
            "score": score_val, "mention": mention,
            "langue": langue, "report_type": report_type,
            "filter_year": year, "filter_region": region, "filter_category": category,
            "periode": faits.get("periode", ""),
            "croissance_pct": faits.get("croissance_globale", 0),
            "generated_at": datetime.utcnow().isoformat(),
            "source": "live",
            # Chantier 4
            "structure": structure,
            # Finitions tonalité / thèmes
            "tonalite_business": tonalite,
            "themes_business": themes,
        }
    except Exception as e:
        logger.warning(f"Génération dynamique NLP échouée: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  v5.3 — INJECTION SECTION NLTK (toujours visible)
# ═══════════════════════════════════════════════════════════════════════════

def _injecter_section_nltk(story, st, faits, langue, nlp_data: dict = None):
    """
    Tente d'afficher NLTK via faits OU via base PostgreSQL.

    FIX BUG #3 : la section NLTK montrait les MÊMES stats dans tous les PDFs
    (569 mots / 22 phrases / 0 mots uniques) car get_nltk_analysis() ne prend
    pas de filtre et renvoyait toujours la dernière analyse globale en DB.
    Désormais : si on a le rapport_complet, on RECALCULE les stats sur CE
    rapport-là, à la volée.

    Finition : enrichit automatiquement avec la tonalité business et les
       thèmes extraits (issus de nlp_data si fourni).
    """
    nltk_data = None

    # 1. NOUVEAU : si on a le rapport_complet, on recalcule les stats
    #    en live → cohérence parfaite avec ce qui est affiché dans le PDF.
    if isinstance(nlp_data, dict) and nlp_data.get("rapport_complet"):
        try:
            rapport_txt = _strip_markdown(nlp_data["rapport_complet"])
            # Calcul autonome (sans NLTK) : robuste et rapide
            phrases = re.split(r"(?<=[\.!?])\s+(?=[A-ZÀ-ÖØ-Þ])", rapport_txt.strip())
            phrases = [p.strip() for p in phrases if len(p.strip()) >= 5]
            mots = re.findall(r"\b[a-zA-Zà-öø-ÿÀ-ÖØ-Þ]+\b", rapport_txt.lower())
            mots_uniques = len(set(mots))
            nb_phrases = len(phrases)
            nb_mots = len(mots)

            # Top 5 mots-clés (TF simple, hors mots-vides minimaux)
            from collections import Counter
            stop_min_fr = {
                "le", "la", "les", "de", "des", "du", "et", "à", "a",
                "en", "un", "une", "que", "qui", "pour", "dans", "sur",
                "par", "se", "ce", "cette", "ces", "son", "sa", "ses",
                "il", "elle", "ils", "elles", "on", "nous", "vous", "est",
                "ont", "été", "avec", "plus", "ne", "pas", "ou", "au", "aux",
            }
            stop_min_en = {
                "the", "a", "an", "of", "and", "to", "in", "for", "on",
                "with", "is", "are", "was", "were", "by", "this", "that",
                "it", "as", "at", "be", "or", "from", "an",
            }
            stop = stop_min_fr if langue == "fr" else stop_min_en
            mots_filtres = [m for m in mots if m not in stop and len(m) >= 4]
            keywords = Counter(mots_filtres).most_common(5)

            nltk_data = {
                "stats": {
                    "nb_phrases": nb_phrases,
                    "nb_mots": nb_mots,
                    "mots_uniques": mots_uniques,
                    "longueur_moyenne_phrase": (nb_mots / nb_phrases) if nb_phrases else 0,
                },
                "keywords": keywords,
                "sentiment": {"pos": 0, "neu": 0, "neg": 0, "compound": 0},
                "entities": {},
                "pos_tags_sample": [],
            }
        except Exception as e:
            logger.warning(f"Recalcul stats NLTK live échoué: {e}")
            nltk_data = None

    # 2. Sinon, tente via faits (si fourni par main.py)
    if not nltk_data and faits and isinstance(faits, dict):
        nltk_data = faits.get("nltk_analysis")

    # 3. Sinon, fallback sur la base (analyse globale - moins précise pour les filtres)
    if not nltk_data:
        try:
            raw = get_nltk_analysis()
            if raw:
                kws = raw.get("keywords") or []
                kws_tuples = []
                if isinstance(kws, list):
                    for k in kws[:15]:
                        if isinstance(k, (list, tuple)) and len(k) >= 2:
                            kws_tuples.append((k[0], k[1]))
                        elif isinstance(k, dict):
                            kws_tuples.append((
                                k.get("mot") or k.get("word") or "",
                                k.get("freq") or k.get("count") or 1
                            ))
                        elif isinstance(k, str):
                            kws_tuples.append((k, 1))

                nb_phrases = raw.get("nb_phrases", 0) or 0
                nb_mots = raw.get("nb_mots", 0) or 0

                nltk_data = {
                    "stats": {
                        "nb_phrases": nb_phrases,
                        "nb_mots": nb_mots,
                        "mots_uniques": raw.get("mots_uniques", 0) or 0,
                        "longueur_moyenne_phrase": (nb_mots / nb_phrases) if nb_phrases else 0,
                    },
                    "keywords": kws_tuples,
                    "sentiment": {
                        "pos": raw.get("sentiment_pos", 0) or 0,
                        "neu": raw.get("sentiment_neu", 0) or 0,
                        "neg": raw.get("sentiment_neg", 0) or 0,
                        "compound": raw.get("sentiment_score", 0) or 0,
                    },
                    "entities": raw.get("entities") or {},
                    "pos_tags_sample": raw.get("pos_tags_sample") or [],
                }
        except Exception as e:
            logger.warning(f"NLTK non disponible: {e}")
            nltk_data = None

    # Finition : injecte tonalité business et thèmes (si dispo dans nlp_data)
    if nltk_data and isinstance(nlp_data, dict):
        if nlp_data.get("tonalite_business"):
            nltk_data["tonalite_business"] = nlp_data["tonalite_business"]
        if nlp_data.get("themes_business"):
            nltk_data["themes"] = nlp_data["themes_business"]
    
    # Si la tonalité n'a pas pu être obtenue via nlp_data mais qu'on a un rapport,
    # on calcule à la volée
    if nltk_data and not nltk_data.get("tonalite_business") \
       and isinstance(nlp_data, dict) and nlp_data.get("rapport_complet"):
        try:
            from src.nlp_nltk import evaluer_tonalite_business, extraire_themes_business
            nltk_data["tonalite_business"] = evaluer_tonalite_business(
                nlp_data["rapport_complet"], langue=langue
            )
            nltk_data["themes"] = extraire_themes_business(
                nlp_data["rapport_complet"], langue=langue, top_n=5
            )
        except Exception as e:
            logger.warning(f"Tonalité business non calculée en live: {e}")

    if nltk_data:
        try:
            elements = creer_section_analyse_linguistique(st, nltk_data, lang=langue)
            story.extend(elements)
        except Exception as e:
            logger.warning(f"Insertion section NLTK échouée: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  FONCTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════

def generer_pdf(faits: dict = None, resultats: dict = None,
                langue: str = "fr", report_type: str = "global",
                year: int = None, region: str = None, category: str = None) -> str:
    """
    Génère un rapport PDF exécutif complet.

    v5.3 : NLP dynamique + markdown nettoyé + NLTK toujours visible.
    LIVRAISON : styles professionnels + images online injectées.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{year}" if year else ""
    suffix += f"_{region}" if region else ""
    suffix += f"_{category}" if category else ""
    filename = f"rapport_business_{langue}_{report_type}{suffix}_{timestamp}.pdf"
    chemin = OUTPUT_DIR / filename

    logger.info(f"Génération du PDF: {chemin}")

    try:
        glob = get_kpi_filtered_summary(year=year, region=region, category=category)

        if _has_filters(year, region, category):
            annual = get_kpi_annual_filtered(year=year, region=region, category=category)
        else:
            annual = get_kpi_annual(year=year)

        regions_data = get_kpi_regions_summary(year=year, region=region, category=category)
        categories_data = get_kpi_categories_summary(year=year, region=region, category=category)
        quarterly = get_kpi_quarterly(year=year, region=region, category=category)
        monthly = get_kpi_monthly(year=year, region=region, category=category)
        segments = get_kpi_segments_summary(year=year, region=region, category=category)
        products = get_kpi_top_products(year=year, region=region, category=category)
        sub_categories = get_kpi_sub_categories(year=year, region=region, category=category, limit=10)
        anomalies_data = get_anomalies(year=year, region=region, category=category)

        # v5.3 — Récupération NLP intelligente (cache + génération dynamique)
        # Chantier 4 — si `resultats` est fourni par main.py, on l'utilise
        #    pour avoir la structure JSON (recos hiérarchisées) + tonalité
        nlp_data = _recuperer_nlp_data_pour_pdf(
            report_type, langue, year, region, category,
            resultats_externes=resultats,
        )

        region_detail = get_kpi_regions(year=year, region=region, category=category)
        logger.info(f"Données PostgreSQL récupérées (filtres: year={year}, region={region}, category={category})")
    except Exception as e:
        logger.warning(f"PostgreSQL indisponible: {e}")
        glob = {}
        annual = regions_data = categories_data = quarterly = monthly = []
        segments = products = anomalies_data = region_detail = sub_categories = []
        nlp_data = {}

    # ── Construction du PDF ──
    doc = SimpleDocTemplate(
        str(chemin), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,    # +marges pour header/footer
        title="NLP Business Report", author="NLP Reporting Pipeline v6.0",
    )

    # LIVRAISON — Styles professionnels avec fallback automatique vers v5.3
    try:
        from src.pdf_design_professional import _styles_professional
        st = _styles_professional()
        logger.info("Styles professionnels chargés (pdf_design_professional)")
    except ImportError:
        st = _styles()
        logger.info("ℹ️ Styles standard v5.3 (pdf_design_professional non trouvé)")

    # ─────────────────────────────────────────────────────────────────
    # v6.0 — HEADER / FOOTER sur chaque page (sauf couverture)
    # ─────────────────────────────────────────────────────────────────
    page_w, page_h = A4

    # Construire l'identifiant de rapport pour le header
    type_short = {
        "global": "Global",
        "by_year": f"Year {year}" if year else "Year",
        "by_region": f"Region · {region}" if region else "Region",
        "by_category": f"Category · {category}" if category else "Category",
    }.get(report_type, "Report")

    def _draw_header_footer(canvas, doc):
        """Dessine header + footer sur chaque page, et la couverture pleine page sur la page 1."""
        canvas.saveState()
        if doc.page == 1:
            # Couverture pleine page — dessinée directement sur le canvas
            _draw_cover_canvas_on_canvas(
                canvas, page_w, page_h,
                report_type, {"year": year, "region": region, "category": category},
                langue, glob,
            )
            canvas.restoreState()
            return

        # ── HEADER ──
        # Filet horizontal subtil
        canvas.setStrokeColor(C.RULE)
        canvas.setLineWidth(0.4)
        canvas.line(2*cm, page_h - 1.4*cm, page_w - 2*cm, page_h - 1.4*cm)

        # Marque de section (eyebrow gauche)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(C.GRIS)
        canvas.drawString(2*cm, page_h - 1.15*cm,
                          f"NLP BUSINESS REPORT  ·  {type_short.upper()}")

        # Période / langue (à droite)
        periode_h = (glob or {}).get("periode", "—")
        canvas.drawRightString(
            page_w - 2*cm, page_h - 1.15*cm,
            f"{'FRANÇAIS' if langue == 'fr' else 'ENGLISH'}  ·  {periode_h}"
        )

        # ── FOOTER ──
        canvas.setStrokeColor(C.RULE)
        canvas.setLineWidth(0.4)
        canvas.line(2*cm, 1.4*cm, page_w - 2*cm, 1.4*cm)

        # Date de génération (gauche)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(C.GRIS)
        canvas.drawString(2*cm, 1.05*cm,
                          f"Généré le {datetime.now().strftime('%d/%m/%Y')}")

        # Pipeline (centre)
        canvas.drawCentredString(
            page_w / 2, 1.05*cm,
            "Spark · NLTK · Mistral · PostgreSQL · FastAPI"
        )

        # Pagination (droite)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawRightString(page_w - 2*cm, 1.05*cm,
                                f"PAGE {doc.page}")
        canvas.restoreState()

    story = []
    filters = {"year": year, "region": region, "category": category}

    # ── Sections ──
    _section_cover(story, st, glob, report_type, filters, langue)

    # Image introductive du résumé exécutif (AVANT le contenu)
    inject_images_to_pdf(story, "resume_executif", st)
    _section_resume_executif(story, st, glob, annual, anomalies_data, nlp_data, filters=filters)

    # KPIs : pas d'image (les cartes sont déjà très visuelles, doublon évité)
    _section_kpis(story, st, glob)

    # Image introductive des analyses graphiques (AVANT les charts, sur nouvelle page)
    story.append(PageBreak())
    inject_images_to_pdf(story, "analyses_graphiques", st)
    _section_graphiques(
        story, st, annual, regions_data, categories_data, quarterly, monthly,
        products, segments, region_detail,
        sub_categories=sub_categories, filters=filters,
    )

    # Image introductive des anomalies (AVANT le contenu, sur nouvelle page)
    story.append(PageBreak())
    inject_images_to_pdf(story, "anomalies", st)
    _section_anomalies(story, st, anomalies_data)

    # Image introductive des recommandations (AVANT le contenu)
    inject_images_to_pdf(story, "recommandations", st)
    # Chantier 4 : recommandations hiérarchisées via structure JSON
    _section_recommandations(
        story, st, glob, regions_data, categories_data, anomalies_data,
        nlp_data=nlp_data, langue=langue,
    )

    # v5.3 — NLTK toujours visible (via faits OU base)
    # Finition — passe nlp_data pour que la section affiche
    #   la tonalité business au lieu du sentiment VADER brut.
    _injecter_section_nltk(story, st, faits, langue, nlp_data=nlp_data)

    _section_nlp(story, st, nlp_data)
    _section_methodologie(story, st)
    _section_conclusion(story, st, glob, anomalies_data)

    doc.build(
        story,
        onFirstPage=_draw_header_footer,
        onLaterPages=_draw_header_footer,
    )
    logger.info(f"PDF généré: {chemin}")

    return str(chemin)


# ═══════════════════════════════════════════════════════════════════════════
#  EXÉCUTION DIRECTE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Génération PDF rapport business")
    parser.add_argument("--langue", choices=["fr", "en"], default="fr")
    parser.add_argument("--type", choices=["global", "by_year", "by_region", "by_category"], default="global")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()

    chemin = generer_pdf(
        langue=args.langue,
        report_type=args.type,
        year=args.year,
        region=args.region,
        category=args.category,
    )
    print(f"\nPDF généré: {chemin}")