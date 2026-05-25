"""Generates custom illustrations for the NLP Reporting presentation.
Style: editorial / scientific / warm - cream palette + vermilion + forest + ocre.
"""
from __future__ import annotations
import math, os, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import path as mpath
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, PathPatch
from PIL import Image, ImageDraw, ImageFilter

OUT = Path("illustrations")
OUT.mkdir(exist_ok=True)

CREAM   = "#FBF7F0"
INK     = "#171513"
VERMI   = "#E63946"
FOREST  = "#1B5E5A"
OCRE    = "#E9A23B"
WARM    = "#B8B0A3"
PAPER   = "#F2EBDD"

random.seed(42); np.random.seed(42)

# 1. COVER HERO
def cover_hero():
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    for x in np.arange(0.3, 12, 0.35):
        for y in np.arange(0.3, 9, 0.35):
            ax.plot(x, y, ".", color=WARM, alpha=0.25, markersize=1.0)
    np.random.seed(7)
    for i in range(180):
        y_start = np.random.uniform(1, 8)
        y_end   = 4.5 + (i % 7 - 3) * 0.45 + np.random.normal(0, 0.08)
        ctrl1_x, ctrl1_y = 4.0, y_start + np.random.normal(0, 0.6)
        ctrl2_x, ctrl2_y = 7.5, (y_start + y_end) / 2 + np.random.normal(0, 0.3)
        verts = [(0.5, y_start), (ctrl1_x, ctrl1_y), (ctrl2_x, ctrl2_y), (11.5, y_end)]
        codes = [mpath.Path.MOVETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]
        p = mpath.Path(verts, codes)
        if i % 23 == 0: c, w, a = VERMI, 0.9, 0.55
        elif i % 17 == 0: c, w, a = FOREST, 0.7, 0.45
        elif i % 11 == 0: c, w, a = OCRE, 0.6, 0.5
        else: c, w, a = INK, 0.35, 0.15
        ax.add_patch(PathPatch(p, facecolor="none", edgecolor=c, lw=w, alpha=a))
    for _ in range(60):
        ax.plot(np.random.uniform(0.2, 1.0), np.random.uniform(0.8, 8.2), ".",
                color=INK, alpha=np.random.uniform(0.2, 0.6),
                markersize=np.random.uniform(0.6, 2.5))
    for i, y in enumerate(np.linspace(2.5, 6.5, 8)):
        ax.plot(11.6, y, "s", color=VERMI if i in (3, 4) else INK, markersize=3.5, alpha=0.85)
    ax.add_patch(Circle((10.5, 4.5), 0.18, facecolor=VERMI, edgecolor="none", zorder=10))
    plt.savefig(OUT / "cover_hero.png", bbox_inches="tight", pad_inches=0, facecolor=CREAM, dpi=200)
    plt.close()

# 2. PIPELINE
def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    stages = [
        (1.8,  "01", "SPARK",      "12 KPIs",          INK),
        (4.0,  "02", "NLTK",       "Faits structures", FOREST),
        (6.2,  "03", "MISTRAL",    "JSON puis prose",  VERMI),
        (8.4,  "04", "SCORE",      "Qualite /100",     OCRE),
        (10.6, "05", "POSTGRESQL", "Source de verite", INK),
    ]
    ax.plot([1.8, 10.6], [4.0, 4.0], color=INK, lw=0.8, zorder=1)
    for x, num, name, sub, col in stages:
        ax.add_patch(Circle((x, 4.0), 0.55, facecolor=CREAM, edgecolor=col, lw=2.0, zorder=3))
        ax.add_patch(Circle((x, 4.0), 0.30, facecolor=col, edgecolor="none", zorder=4))
        ax.text(x, 4.0, num, ha="center", va="center", fontsize=11, fontweight="bold",
                color=CREAM, family="monospace", zorder=5)
        ax.text(x, 5.2, name, ha="center", va="center", fontsize=14, fontweight="bold",
                color=INK, family="serif")
        ax.text(x, 2.85, sub, ha="center", va="center", fontsize=10,
                color=WARM, family="sans-serif", style="italic")
    outputs = [(2.5, "PDF"), (7.0, "DASHBOARD"), (11.5, "CHATBOT")]
    for ox, label in outputs:
        ax.plot([10.6, ox], [4.0, 1.2], color=WARM, lw=0.6, ls="--", zorder=0)
        ax.add_patch(FancyBboxPatch((ox-1.0, 0.5), 2.0, 0.7,
                     boxstyle="round,pad=0.05,rounding_size=0.1",
                     facecolor=INK, edgecolor="none"))
        ax.text(ox, 0.85, label, ha="center", va="center", fontsize=11,
                color=CREAM, family="monospace", fontweight="bold")
    ax.text(0.3, 6.5, "P I P E L I N E", fontsize=9, color=VERMI,
            family="monospace", fontweight="bold")
    ax.text(0.3, 6.15, "De la donnee brute au rapport business", fontsize=14,
            color=INK, family="serif", style="italic")
    plt.savefig(OUT / "pipeline.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

# 3. HALLUCINATION COMPARE
def hallucination_compare():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=200)
    fig.patch.set_facecolor(CREAM)
    ax = axes[0]
    ax.set_facecolor(CREAM); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    t = np.linspace(0, 1, 200)
    x = 1 + 8 * t
    y = 4 + np.sin(t * 18) * 0.7 + np.cos(t * 9) * 0.4 + np.random.normal(0, 0.05, 200)
    ax.plot(x, y, color=VERMI, lw=2.2, alpha=0.9)
    ax.plot([1], [4], "o", color=INK, markersize=14)
    ax.plot([9], [y[-1]], "X", color=VERMI, markersize=22)
    ax.text(1, 3.2, "FAITS", ha="center", fontsize=10, color=INK, family="monospace", fontweight="bold")
    ax.text(9, y[-1]-0.8, "+35%", ha="center", fontsize=14, color=VERMI, family="serif", fontweight="bold", style="italic")
    ax.text(9, y[-1]-1.3, "(faux)", ha="center", fontsize=9, color=VERMI, family="monospace")
    ax.text(5, 6.3, "DIRECT . LLM LIBRE", ha="center", fontsize=10, color=WARM, family="monospace")
    ax.text(5, 5.7, "Hallucination", ha="center", fontsize=20, color=INK, family="serif", fontweight="bold", style="italic")

    ax = axes[1]
    ax.set_facecolor(CREAM); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.plot([1, 4.5], [4, 4], color=FOREST, lw=2.2)
    ax.plot([5.5, 9], [4, 4], color=FOREST, lw=2.2)
    ax.plot([1], [4], "o", color=INK, markersize=14)
    ax.add_patch(Rectangle((4.5, 3.5), 1.0, 1.0, facecolor=CREAM, edgecolor=FOREST, lw=2))
    ax.text(5.0, 4, "JSON\nVALIDE", ha="center", va="center", fontsize=8, color=FOREST, family="monospace", fontweight="bold")
    ax.plot([9], [4], "o", color=FOREST, markersize=18)
    ax.text(9, 4, "OK", ha="center", va="center", fontsize=11, color=CREAM, fontweight="bold")
    ax.text(1, 3.2, "FAITS", ha="center", fontsize=10, color=INK, family="monospace", fontweight="bold")
    ax.text(9, 3.2, "+49.5%", ha="center", fontsize=14, color=FOREST, family="serif", fontweight="bold", style="italic")
    ax.text(9, 2.7, "(exact)", ha="center", fontsize=9, color=FOREST, family="monospace")
    ax.text(5, 6.3, "DEUX ETAPES . LLM ENCADRE", ha="center", fontsize=10, color=WARM, family="monospace")
    ax.text(5, 5.7, "Garantie", ha="center", fontsize=20, color=INK, family="serif", fontweight="bold", style="italic")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.15)
    plt.savefig(OUT / "hallucination.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

# 4. SCORE GAUGE
def score_gauge():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM); ax.axis("off")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    weights = [25, 25, 15, 15, 10, 10]
    colors  = [VERMI, FOREST, OCRE, INK, VERMI, FOREST]
    labels  = ["Couverture\ndes faits", "Ancrage\nnumerique",
               "Recos\npresentes", "Clarte", "Anti-\nrepetition", "Ton\nbusiness"]
    total = sum(weights)
    start = 90
    for w, c, lab in zip(weights, colors, labels):
        end = start - (w / total) * 360
        theta = np.linspace(np.radians(end), np.radians(start), 60)
        r_out, r_in = 1.15, 0.85
        x_out, y_out = r_out * np.cos(theta), r_out * np.sin(theta)
        x_in, y_in = r_in * np.cos(theta[::-1]), r_in * np.sin(theta[::-1])
        verts = list(zip(np.concatenate([x_out, x_in]), np.concatenate([y_out, y_in])))
        ax.add_patch(plt.Polygon(verts, closed=True, facecolor=c, edgecolor=CREAM, lw=2.5))
        mid = np.radians((start + end) / 2)
        lx, ly = 1.4 * np.cos(mid), 1.4 * np.sin(mid)
        ax.text(lx, ly, lab, ha="center", va="center", fontsize=8.5, color=INK, family="sans-serif")
        rx, ry = 1.0 * np.cos(mid), 1.0 * np.sin(mid)
        ax.text(rx, ry, str(w), ha="center", va="center", fontsize=11, color=CREAM,
                family="monospace", fontweight="bold")
        start = end
    ax.add_patch(Circle((0, 0), 0.75, facecolor=CREAM, edgecolor="none"))
    ax.text(0, 0.07, "87", ha="center", va="center", fontsize=72, color=INK, family="serif", fontweight="bold")
    ax.text(0, -0.32, "/ 100", ha="center", va="center", fontsize=14, color=WARM, family="monospace")
    plt.savefig(OUT / "score_gauge.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

# 5. DASHBOARD MOCK
def dashboard_mock():
    fig, ax = plt.subplots(figsize=(13, 7), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    ax.text(0.5, 6.5, "D A S H B O A R D", fontsize=9, color=VERMI, family="monospace", fontweight="bold")
    ax.text(0.5, 6.05, "Business Intelligence", fontsize=22, color=INK, family="serif", style="italic")
    for i, lbl in enumerate(["TOUTES ANNEES", "TOUTES REGIONS", "TOUTES CATEGORIES", "FR/EN"]):
        x = 0.5 + i * 1.85
        ax.add_patch(FancyBboxPatch((x, 5.45), 1.7, 0.35,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     facecolor=CREAM, edgecolor=WARM, lw=0.8))
        ax.text(x + 0.85, 5.625, lbl, ha="center", va="center", fontsize=7.5, color=INK, family="monospace")
    ax.add_patch(FancyBboxPatch((8.0, 5.45), 1.7, 0.35,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 facecolor=VERMI, edgecolor="none"))
    ax.text(8.85, 5.625, "EXPORT PDF", ha="center", va="center", fontsize=7.5,
            color=CREAM, family="monospace", fontweight="bold")
    kpis = [
        ("$2.2M", "CHIFFRE D'AFFAIRES", "+49.5%", VERMI),
        ("+49.5%", "CROISSANCE",        "2015-18", FOREST),
        ("4 842", "COMMANDES",          None, INK),
        ("$462",  "PANIER MOYEN",       None, OCRE),
        ("West",  "TOP REGION",         None, FOREST),
        ("793",   "CLIENTS",            None, VERMI),
    ]
    for i, (v, l, b, c) in enumerate(kpis):
        col = i % 3; row = i // 3
        x = 0.5 + col * 4.05; y = 3.0 - row * 2.05
        ax.add_patch(FancyBboxPatch((x, y), 3.85, 1.85,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor=PAPER, edgecolor="none"))
        ax.add_patch(Circle((x + 0.3, y + 1.55), 0.07, facecolor=c, edgecolor="none"))
        ax.text(x + 0.5, y + 1.5, l, fontsize=7.5, color=WARM, family="monospace")
        ax.text(x + 0.3, y + 0.85, v, fontsize=28, color=INK, family="serif", fontweight="bold")
        if b:
            ax.add_patch(FancyBboxPatch((x + 2.85, y + 1.45), 0.9, 0.28,
                         boxstyle="round,pad=0.01,rounding_size=0.05",
                         facecolor=c, edgecolor="none"))
            ax.text(x + 3.3, y + 1.59, b, ha="center", va="center", fontsize=7.5,
                    color=CREAM, family="monospace", fontweight="bold")
    plt.savefig(OUT / "dashboard_mock.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

# 6. POSTGRES HUB
def postgres_hub():
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.set_aspect("equal")
    cx, cy = 5, 4
    for r, a in [(3.5, 0.10), (3.0, 0.13), (2.5, 0.16), (2.0, 0.20)]:
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=INK, lw=0.5, alpha=a))
    ax.add_patch(Circle((cx, cy), 1.1, facecolor=INK, edgecolor="none", zorder=5))
    ax.text(cx, cy + 0.2, "PostgreSQL", ha="center", va="center", fontsize=14,
            color=CREAM, family="serif", fontweight="bold", zorder=6)
    ax.text(cx, cy - 0.25, "12 tables", ha="center", va="center", fontsize=9,
            color=CREAM, family="monospace", zorder=6)
    ax.text(cx, cy - 0.55, "source unique", ha="center", va="center", fontsize=8,
            color=OCRE, family="monospace", style="italic", zorder=6)
    consumers = [
        (1.2, 6.5, "PDF",      VERMI),
        (8.8, 6.5, "DASHBOARD", FOREST),
        (1.2, 1.5, "API REST", OCRE),
        (8.8, 1.5, "CHATBOT",  VERMI),
    ]
    for x, y, label, c in consumers:
        ax.plot([cx, x], [cy, y], color=WARM, lw=0.6, ls="-", zorder=1)
        for t in [0.3, 0.55, 0.8]:
            ax.plot(cx + (x - cx) * t, cy + (y - cy) * t, ".",
                    color=c, markersize=4, alpha=0.7, zorder=2)
        ax.add_patch(Circle((x, y), 0.7, facecolor=CREAM, edgecolor=c, lw=2, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=10, color=INK,
                family="monospace", fontweight="bold", zorder=4)
    plt.savefig(OUT / "postgres_hub.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

# 7. CHATBOT FLOW
def chatbot_flow():
    fig, ax = plt.subplots(figsize=(13, 7), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    steps = [
        "Question utilisateur",
        "Detection d'injection",
        "Detection d'intention",
        "Contexte PostgreSQL",
        "Embedding 384d",
        "ChromaDB top 5",
        "Appel Mistral",
        "Reponse 3-4 phrases",
    ]
    for i, st in enumerate(steps):
        y = 6.3 - i * 0.7
        ax.add_patch(Circle((0.7, y), 0.18, facecolor=INK, edgecolor="none"))
        ax.text(0.7, y, str(i+1), ha="center", va="center", fontsize=9,
                color=CREAM, family="monospace", fontweight="bold")
        ax.text(1.1, y, st, va="center", fontsize=11, color=INK, family="sans-serif")
        if i < len(steps) - 1:
            ax.plot([0.7, 0.7], [y - 0.18, y - 0.52], color=WARM, lw=0.6)
    ax.add_patch(FancyBboxPatch((6.5, 1), 6.0, 5.5,
                 boxstyle="round,pad=0.05,rounding_size=0.12",
                 facecolor=PAPER, edgecolor=WARM, lw=0.8))
    ax.text(6.8, 6.2, "A S S I S T A N T   I A", fontsize=8, color=VERMI,
            family="monospace", fontweight="bold")
    ax.add_patch(FancyBboxPatch((7.0, 4.8), 4.8, 0.9,
                 boxstyle="round,pad=0.05,rounding_size=0.12",
                 facecolor=INK, edgecolor="none"))
    ax.text(7.2, 5.25, "Quelle est la pire region en 2017 ?",
            va="center", fontsize=11, color=CREAM, family="sans-serif")
    ax.add_patch(Circle((7.2, 3.7), 0.22, facecolor=VERMI, edgecolor="none"))
    ax.text(7.2, 3.7, "AI", ha="center", va="center", fontsize=8, color=CREAM,
            family="monospace", fontweight="bold")
    ax.add_patch(FancyBboxPatch((7.6, 2.0), 4.6, 2.0,
                 boxstyle="round,pad=0.05,rounding_size=0.12",
                 facecolor=CREAM, edgecolor=WARM, lw=0.5))
    txt = ("Central affiche la plus faible\n"
           "progression en 2017 avec -8%\n"
           "vs 2016. Le segment Corporate\n"
           "y recule de 14% - signal de\n"
           "vigilance prioritaire.")
    ax.text(7.8, 3.0, txt, va="center", fontsize=10, color=INK, family="sans-serif")
    plt.savefig(OUT / "chatbot.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

# 8. INTENTIONS
def intentions_diagram():
    fig, ax = plt.subplots(figsize=(13, 7), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    intents = [
        ("01", "TENDANCE",       "Decrit ce qui se passe",            FOREST),
        ("02", "ANOMALIE",       "Signale un ecart inhabituel",       VERMI),
        ("03", "OPPORTUNITE",    "Identifie un levier de croissance", OCRE),
        ("04", "RISQUE",         "Alerte sur un point de vigilance",  VERMI),
        ("05", "RECOMMANDATION", "Propose une action concrete",       FOREST),
        ("06", "CONTEXTE",       "Le reste du texte",                 WARM),
    ]
    for i, (num, name, desc, c) in enumerate(intents):
        col = i % 3; row = i // 3
        x = 0.5 + col * 4.15
        y = 4.0 - row * 3.0
        ax.text(x + 3.4, y + 1.7, num, fontsize=72, color=c, alpha=0.18,
                family="serif", fontweight="bold", ha="right", va="top")
        ax.text(x + 0.1, y + 1.5, name, fontsize=12, color=INK,
                family="monospace", fontweight="bold")
        ax.plot([x + 0.1, x + 0.7], [y + 1.25, y + 1.25], color=c, lw=1.5)
        ax.text(x + 0.1, y + 0.7, desc, fontsize=11, color=INK,
                family="sans-serif", style="italic")
    plt.savefig(OUT / "intentions.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

# 9. RECO CARD
def reco_card():
    fig, ax = plt.subplots(figsize=(13, 5), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)
    ax.add_patch(Rectangle((0.5, 0.5), 0.18, 4.0, facecolor=VERMI, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((0.68, 0.5), 11.8, 4.0,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=PAPER, edgecolor="none"))
    badges = [("HAUTE", VERMI), ("STRATEGIQUE", INK), ("CONFIANCE ELEVEE", FOREST)]
    for i, (lab, c) in enumerate(badges):
        x = 1.0 + i * 3.0
        ax.add_patch(FancyBboxPatch((x, 3.7), 2.7, 0.45,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor=CREAM, edgecolor=c, lw=1.2))
        ax.text(x + 1.35, 3.925, lab, ha="center", va="center", fontsize=8.5,
                color=c, family="monospace", fontweight="bold")
    ax.text(1.0, 3.0, "Renforcer la presence commerciale en region West",
            fontsize=20, color=INK, family="serif", fontweight="bold")
    ax.text(1.0, 2.0, "J U S T I F I C A T I O N", fontsize=8, color=WARM,
            family="monospace")
    ax.text(1.0, 1.45, "West genere 32 % du CA total et affiche une croissance YoY de +52 %.\n"
            "La region concentre les segments Corporate et Consumer a forte marge.",
            fontsize=12, color=INK, family="sans-serif", style="italic")
    plt.savefig(OUT / "reco_card.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

# 10. CHART
def chart_sample():
    fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    np.random.seed(3)
    months = np.arange(48)
    base = 10 + months * 0.45
    seasonal = 8 * np.sin(months * np.pi / 6) + 4 * np.cos(months * np.pi / 12)
    noise = np.random.normal(0, 2.5, 48)
    sales = np.maximum(base + seasonal + noise, 2)
    moving = np.convolve(sales, np.ones(3) / 3, mode="same")
    ax.fill_between(months, sales, alpha=0.10, color=VERMI)
    ax.plot(months, sales, color=VERMI, lw=1.4, alpha=0.7, marker="o", markersize=3)
    ax.plot(months, moving, color=INK, lw=2.2, label="Moyenne mobile 3M")
    pk = int(np.argmax(sales))
    ax.plot(pk, sales[pk], "o", color=INK, markersize=11, mfc=CREAM, mec=INK, mew=2)
    ax.annotate("Pic Nov 2018", (pk, sales[pk]),
                xytext=(pk-8, sales[pk]+4), fontsize=10, color=INK,
                family="serif", style="italic",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    ax.set_xticks([0, 12, 24, 36, 47])
    ax.set_xticklabels(["2015", "2016", "2017", "2018", ""], fontsize=10,
                       color=WARM, family="monospace")
    ax.set_yticks([])
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(WARM); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", colors=WARM, length=0)
    ax.text(0, ax.get_ylim()[1] * 0.95, "EVOLUTION MENSUELLE",
            fontsize=9, color=VERMI, family="monospace", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "chart_sample.png", bbox_inches="tight", pad_inches=0.15, facecolor=CREAM, dpi=200)
    plt.close()

# 11. PAPER TEXTURE
def paper_texture():
    img = Image.new("RGB", (1920, 1080), CREAM)
    draw = ImageDraw.Draw(img)
    rng = random.Random(11)
    for _ in range(8000):
        x = rng.randint(0, 1919); y = rng.randint(0, 1079)
        shade = rng.randint(218, 240)
        draw.point((x, y), fill=(shade, shade - 3, shade - 12))
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(OUT / "paper_bg.png", "PNG", quality=92)

# 12. BILAN NUMBERS
def bilan_numbers():
    fig, ax = plt.subplots(figsize=(13, 7), dpi=200)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    items = [
        ("12",  "KPIs",        "calcules par Spark",          VERMI),
        ("4",   "CHANTIERS",   "NLP mesurables",              FOREST),
        ("12",  "TABLES",      "source unique de verite",     INK),
        ("20+", "ENDPOINTS",   "API documentes",              OCRE),
        ("25",  "PAGES",       "PDF a la demande",            VERMI),
        ("100", "% COHERENCE", "entre tous les canaux",       FOREST),
    ]
    for i, (n, l1, l2, c) in enumerate(items):
        col = i % 3; row = i // 3
        x = 0.4 + col * 4.2; y = 3.7 - row * 3.3
        ax.text(x, y + 0.5, n, fontsize=110, color=c, alpha=0.95,
                family="serif", fontweight="bold")
        ax.text(x + 0.2, y + 0.4, l1, fontsize=11, color=INK,
                family="monospace", fontweight="bold")
        ax.text(x + 0.2, y + 0.05, l2, fontsize=10, color=WARM,
                family="sans-serif", style="italic")
    plt.savefig(OUT / "bilan.png", bbox_inches="tight", pad_inches=0.1, facecolor=CREAM, dpi=200)
    plt.close()

if __name__ == "__main__":
    funcs = [cover_hero, pipeline_diagram, hallucination_compare, score_gauge,
             dashboard_mock, postgres_hub, chatbot_flow, intentions_diagram,
             reco_card, chart_sample, paper_texture, bilan_numbers]
    for f in funcs:
        print("  > " + f.__name__)
        f()
    print("Done.")
