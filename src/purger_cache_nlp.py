"""
purger_cache_nlp.py - Utilitaire de purge du cache NLP en BDD
==============================================================

Role
----
Script ponctuel a executer apres une modification de la logique de
generation NLP (par exemple un changement de prompt Mistral ou un
correctif sur la croissance YoY). Permet de detecter et purger les
rapports en cache qui contiennent des chiffres devenus obsoletes.

Pourquoi ce script existe
-------------------------
Le pipeline cache les rapports NLP dans la table `rapports_nlp` pour
eviter de regenerer chaque fois. Mais si la logique applicative change
(ex : passage de QoQ a YoY, gestion du None sur premiere annee), les
anciens textes en cache contiennent encore des chiffres faux. Tant
qu'on ne purge pas, le dashboard et le PDF affichent les anciennes
versions.

Mode d'emploi
-------------
    python src/purger_cache_nlp.py            (mode dry-run : compte seulement)
    python src/purger_cache_nlp.py --confirm  (purge effective)

Apres la purge, lancer `python main.py` pour regenerer les 12 rapports.

Heuristiques de detection
-------------------------
- Bullet contenant "+0.0%" ou "0.0 %" (indice d'un calcul de croissance
  errone sur un rapport mono-annee).
- Rapport contenant "stabilite globale" / "global stability" (texte
  type quand cg=0.0 etait envoye a Mistral).
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Effectue reellement le TRUNCATE",
    )
    args = parser.parse_args()

    try:
        from src.db import get_cursor
    except ImportError:
        from db import get_cursor

    print("=" * 60)
    print("  AUDIT DU CACHE rapports_nlp")
    print("=" * 60)

    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM rapports_nlp;")
        total = cur.fetchone()["n"]

        cur.execute(
            r"""SELECT COUNT(*) AS n FROM rapports_nlp
                WHERE resume_bullet1 ~ '\+?0[.,]0\s?%';"""
        )
        suspects_pct = cur.fetchone()["n"]

        cur.execute(
            """SELECT COUNT(*) AS n FROM rapports_nlp
               WHERE rapport_complet ILIKE '%stabilite globale%'
                  OR rapport_complet ILIKE '%global stability%';"""
        )
        suspects_text = cur.fetchone()["n"]

        cur.execute(
            """SELECT COUNT(*) AS n FROM rapports_nlp
               WHERE report_type = 'by_year' AND filter_year IS NOT NULL;"""
        )
        mono_year = cur.fetchone()["n"]

    print(f"  Total de rapports en cache  : {total}")
    print(f"  Rapports filtres mono-annee : {mono_year}")
    print(f"  Bullets avec '+0.0%'        : {suspects_pct}")
    print(f"  Rapports avec 'stabilite globale' : {suspects_text}")
    print()

    a_purger = max(suspects_pct, suspects_text)
    if a_purger == 0:
        print("[OK] Aucun cache suspect detecte. Pas de purge necessaire.")
        return 0

    print(f"[WARN] Au moins {a_purger} rapport(s) suspect(s) en cache.")
    print()

    if not args.confirm:
        print("Mode dry-run - aucune modification effectuee.")
        print("Pour purger : python src/purger_cache_nlp.py --confirm")
        return 0

    print("Purge en cours...")
    with get_cursor() as cur:
        cur.execute("TRUNCATE TABLE rapports_nlp RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE TABLE nltk_analysis RESTART IDENTITY CASCADE;")
    print("[OK] Cache rapports_nlp et nltk_analysis purges.")
    print()
    print("Etape suivante : regenerer les rapports avec la logique courante :")
    print("   python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
