"""
pdf_images_online.py - Gestionnaire d'images pour les PDFs
============================================================

Role
----
Charge les illustrations injectees au debut de certaines sections du
PDF (resume executif, analyses graphiques, anomalies, recommandations,
methodologie). Source : photos Unsplash verifiees visuellement, mises
en cache local apres premier telechargement.

Pourquoi ce module
------------------
- Permet d'enrichir visuellement les livrables sans embarquer
  d'assets dans le repo.
- Cache local : un seul telechargement par photo, reutilise ensuite.
- Fallback vectoriel matplotlib si le reseau est indisponible.

API publique
------------
- inject_images_to_pdf(story, section_name, st)
    Injecte une image dans la `story` ReportLab pour la section donnee.
- get_cover_photo_bytes()
    Renvoie les bytes JPEG de la photo de couverture (utilisee par
    generate_pdf._draw_cover_canvas_on_canvas).

Sections supportees
-------------------
cover, resume_executif, kpis, analyses_graphiques, anomalies,
recommandations, methodologie (+ alias EN equivalents).

Cache
-----
Stocke par defaut dans ~/.cache/nlp_reporting_images (modifiable via
la variable d'env PDF_IMAGES_CACHE).
"""

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Optional

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from reportlab.platypus import Image as RLImage, Spacer, Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

logger = logging.getLogger(__name__)


CACHE_DIR = Path(os.environ.get(
    "PDF_IMAGES_CACHE",
    str(Path.home() / ".cache" / "nlp_reporting_images"),
))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


VERIFIED_PHOTOS = {
    # Couverture : entrepot
    "cover": "https://images.unsplash.com/photo-1553413077-190dd305871c?w=1600&q=80",
    # Resume executif : reunion business
    "resume_executif": "https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=1200&q=80",
    # KPIs : dashboard analytics
    "kpis": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80",
    "kpi": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80",
    # Analyses : laptop avec courbes
    "analyses_graphiques": "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=1200&q=80",
    "charts": "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=1200&q=80",
    # Anomalies : examen de documents
    "anomalies": "https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=1200&q=80",
    "anomalies_alertes": "https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=1200&q=80",
    "alerts": "https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=1200&q=80",
    # Recommandations : reunion strategique
    "recommandations": "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=1200&q=80",
    "recommendations": "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=1200&q=80",
    # Methodologie : code sur ecran
    "methodologie": "https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=1200&q=80",
    "methodology": "https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=1200&q=80",
}


def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{h}.jpg"


def _download_image(url: str, timeout: int = 8) -> Optional[bytes]:
    """Telecharge (ou lit depuis cache). None en cas d'echec."""
    cache_file = _cache_path(url)
    if cache_file.exists() and cache_file.stat().st_size > 1000:
        try:
            return cache_file.read_bytes()
        except Exception:
            pass

    try:
        logger.info(f"Telechargement: {url[:60]}...")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.content
        if len(data) < 1000:
            logger.warning(f"Image trop petite ({len(data)} bytes), ignoree")
            return None
        try:
            cache_file.write_bytes(data)
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")
        return data
    except Exception as e:
        logger.warning(f"Echec telechargement {url[:50]}: {e}")
        return None


def _fallback_vector(section_name: str) -> bytes:
    """Illustration vectorielle de secours (PNG bytes)."""
    BLUE = "#1A5FA8"
    GREEN = "#639922"
    GREY = "#888780"

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 26), 100, 4, facecolor=BLUE, edgecolor="none"))
    ax.add_patch(Rectangle((0, 24), 20, 2, facecolor=GREEN, edgecolor="none"))

    label = (section_name or "section").replace("_", " ").upper()
    ax.text(2, 16, label, fontsize=14, color=BLUE, fontweight="bold")
    ax.text(2, 11, "Pipeline analytique automatise",
            fontsize=9, color=GREY, style="italic")

    for i, h in enumerate([3, 5, 4, 7, 6, 8]):
        ax.add_patch(Rectangle(
            (60 + i * 5, 4), 3, h,
            facecolor=BLUE if i % 2 == 0 else GREEN,
            alpha=0.7, edgecolor="none",
        ))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def inject_images_to_pdf(story: list, section_name: str, st: dict = None) -> None:
    """Injecte une photo (ou fallback vectoriel) pour la section demandee."""
    url = VERIFIED_PHOTOS.get(section_name)
    if not url:
        logger.debug(f"Pas d'image pour la section '{section_name}'")
        return

    data = _download_image(url)
    if data is None:
        logger.info(f"Fallback vectoriel pour '{section_name}'")
        data = _fallback_vector(section_name)

    try:
        img = RLImage(io.BytesIO(data), width=16 * cm, height=6.0 * cm)
        story.append(Spacer(1, 4))
        story.append(img)
        story.append(Spacer(1, 10))
    except Exception as e:
        logger.warning(f"Echec injection image '{section_name}': {e}")


def get_cover_photo_bytes() -> Optional[bytes]:
    """Bytes JPEG de la photo de couverture (utilisee en fond de page 1)."""
    return _download_image(VERIFIED_PHOTOS["cover"])


# Retrocompatibilite

class OnlineImages:
    """Stub pour compatibilite ascendante."""
    RESUME = {"header": VERIFIED_PHOTOS["resume_executif"]}
    KPI = {"dashboard": VERIFIED_PHOTOS["kpis"]}
    CHARTS = {"analytics": VERIFIED_PHOTOS["analyses_graphiques"]}
    ALERTS = {"warning": VERIFIED_PHOTOS["anomalies"]}
    RECOMMENDATIONS = {"strategy": VERIFIED_PHOTOS["recommandations"]}
    METHODOLOGY = {"process": VERIFIED_PHOTOS["methodologie"]}
    ICONS = {}


class OnlineImageManager:
    """Compat ascendante."""

    def __init__(self, timeout=10):
        self.timeout = timeout
        self.failed_urls = set()

    def load_image(self, url, width=10 * cm, height=None, fallback_emoji="."):
        data = _download_image(url, timeout=self.timeout)
        if data is None:
            return None
        try:
            return RLImage(io.BytesIO(data), width=width, height=height)
        except Exception:
            return None

    def load_images_batch(self, urls, width=10 * cm):
        return {name: self.load_image(url, width=width) for name, url in urls.items()}


def get_online_image(section, image_type="header"):
    return VERIFIED_PHOTOS.get(section, "")


def add_section_header_image(story, section_name, title, st=None):
    inject_images_to_pdf(story, section_name, st)
    if st and title:
        story.append(Paragraph(f"<b>{title.upper()}</b>", st.get("h1")))
        story.append(Spacer(1, 8))


def add_kpi_cards_with_images(story, kpis, st=None):
    if not kpis:
        return
    cards = []
    for kpi in kpis[:3]:
        if st:
            value_para = Paragraph(kpi.get("value", "N/A"), st.get("kpi_val", {}))
            label_para = Paragraph(kpi.get("label", ""), st.get("kpi_lbl", {}))
        else:
            value_para = Paragraph(kpi.get("value", "N/A"), {})
            label_para = Paragraph(kpi.get("label", ""), {})
        cards.append([value_para, Spacer(1, 4), label_para])

    table = Table([cards], colWidths=[5 * cm] * len(cards))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E6F1FB")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEABOVE", (0, 0), (-1, 0), 2, colors.HexColor("#1A5FA8")),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))


def add_alert_with_icon(story, alert_text, level="warning", st=None):
    color_map = {
        "warning": "#1A5FA8",
        "success": "#639922",
        "error": "#E24B4A",
        "info": "#0D3F6E",
    }
    bar_color = colors.HexColor(color_map.get(level, "#888780"))

    if st:
        text_para = Paragraph(alert_text, st.get("body", {}))
    else:
        text_para = Paragraph(alert_text, {})

    table = Table([[text_para]], colWidths=[15 * cm])
    table.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 3, bar_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 6))


if __name__ == "__main__":
    print("Gestionnaire d'images Unsplash")
    print(f"Cache: {CACHE_DIR}")
    print(f"{len(set(VERIFIED_PHOTOS.values()))} photos verifiees")
    print("\nPrechargement du cache...")
    for name, url in VERIFIED_PHOTOS.items():
        data = _download_image(url)
        status = "[OK]" if data else "[KO]"
        size = f"({len(data)/1024:.0f}kB)" if data else ""
        print(f"   {status} {name} {size}")
