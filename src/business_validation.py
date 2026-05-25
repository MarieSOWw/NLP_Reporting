"""
business_validation.py - Validation metier automatique des KPIs
==================================================================

Role
----
Verifie la coherence interne des KPIs en BDD. Permet de detecter
silencieusement des incoherences (CA global different de la somme des
annees, panier moyen mal calcule, valeur negative impossible, etc.).

Utilise par
-----------
- Endpoint /api/validation/business (api.py)
- Section optionnelle du dashboard pour afficher un badge "donnees OK"
- Smoke test apres execution du pipeline

Principe
--------
Chaque check renvoie un dict standardise :
    {
        "name": str,
        "status": "ok" | "warning" | "error",
        "expected": float | None,
        "actual": float | None,
        "ecart_pct": float | None,
        "message": str,
    }

Tolerance par defaut : 1% d'ecart. Au-dela on emet un warning,
au-dela de 5% c'est une erreur.

Checks implementes
------------------
1. CA global = somme des CA annuels
2. CA annuel = somme des CA trimestriels par annee
3. CA par region somme = CA total (sur ventes_detail)
4. Panier moyen = CA / Nb_Commandes (sur kpi_global)
5. Aucune valeur de ventes negative dans ventes_detail
6. Aucune ligne ventes_detail avec date / region / categorie null
7. Croissance YoY coherente (sign agree entre annees)
"""

import logging
from typing import List, Dict, Optional

from src.db import get_cursor

logger = logging.getLogger(__name__)


TOLERANCE_WARN = 1.0
TOLERANCE_ERR = 5.0


def _check(name: str, expected: Optional[float], actual: Optional[float],
           message_ok: str, message_ko: str,
           absolute_tolerance: bool = False) -> Dict:
    """Helper : construit un dict de resultat standardise."""
    if expected is None or actual is None:
        return {
            "name": name,
            "status": "warning",
            "expected": expected,
            "actual": actual,
            "ecart_pct": None,
            "message": f"Donnees manquantes pour le check (expected={expected}, actual={actual})",
        }

    diff = abs(actual - expected)
    if absolute_tolerance:
        ecart_pct = diff
    else:
        ecart_pct = (diff / abs(expected) * 100) if expected != 0 else (0 if actual == 0 else 100)

    if ecart_pct <= TOLERANCE_WARN:
        status = "ok"
        msg = message_ok
    elif ecart_pct <= TOLERANCE_ERR:
        status = "warning"
        msg = f"{message_ko} (ecart {ecart_pct:.2f}%)"
    else:
        status = "error"
        msg = f"{message_ko} (ecart {ecart_pct:.2f}%)"

    return {
        "name": name,
        "status": status,
        "expected": round(expected, 2) if expected is not None else None,
        "actual": round(actual, 2) if actual is not None else None,
        "ecart_pct": round(ecart_pct, 2),
        "message": msg,
    }


def check_ca_global_vs_annuel(cur) -> Dict:
    """CA global doit egaler la somme des CA annuels."""
    cur.execute("SELECT ca_total FROM kpi_global ORDER BY created_at DESC LIMIT 1;")
    row = cur.fetchone()
    ca_global = float(row["ca_total"]) if row and row.get("ca_total") else None

    cur.execute("SELECT COALESCE(SUM(ca_annuel), 0) AS s FROM kpi_annuel;")
    sum_annuel = float(cur.fetchone()["s"])

    return _check(
        "CA global = somme(CA annuels)",
        expected=ca_global,
        actual=sum_annuel,
        message_ok="CA global coherent avec la somme des CA annuels.",
        message_ko="CA global different de la somme des CA annuels.",
    )


def check_ca_annuel_vs_trimestriel(cur) -> Dict:
    """Pour chaque annee, CA annuel doit egaler la somme des trimestres (region_trim)."""
    cur.execute(
        """
        SELECT k.annee,
               k.ca_annuel,
               COALESCE(SUM(rt.ventes_totales), 0) AS sum_trim
        FROM kpi_annuel k
        LEFT JOIN kpi_region_trim rt ON rt.annee = k.annee
        GROUP BY k.annee, k.ca_annuel
        ORDER BY k.annee;
        """
    )
    rows = cur.fetchall()

    if not rows:
        return {
            "name": "CA annuel = somme(CA trimestriels)",
            "status": "warning",
            "expected": None, "actual": None, "ecart_pct": None,
            "message": "Aucune donnee kpi_annuel a verifier.",
        }

    ecarts = []
    for r in rows:
        ca_a = float(r.get("ca_annuel") or 0)
        sum_t = float(r.get("sum_trim") or 0)
        if ca_a > 0:
            pct = abs(ca_a - sum_t) / ca_a * 100
            ecarts.append((int(r["annee"]), ca_a, sum_t, pct))

    if not ecarts:
        return {
            "name": "CA annuel = somme(CA trimestriels)",
            "status": "warning",
            "expected": None, "actual": None, "ecart_pct": None,
            "message": "Aucun CA annuel non nul a verifier.",
        }

    ecart_max = max(ecarts, key=lambda x: x[3])
    annee, ca_a, sum_t, pct = ecart_max

    if pct <= TOLERANCE_WARN:
        status, msg = "ok", f"CA annuel coherent avec les trimestres (ecart max {pct:.2f}% sur {annee})."
    elif pct <= TOLERANCE_ERR:
        status, msg = "warning", f"Ecart {pct:.2f}% en {annee} entre CA annuel ({ca_a:,.0f}) et somme trimestres ({sum_t:,.0f})."
    else:
        status, msg = "error", f"Ecart important ({pct:.2f}%) en {annee} : CA annuel={ca_a:,.0f}, somme trimestres={sum_t:,.0f}."

    return {
        "name": "CA annuel = somme(CA trimestriels)",
        "status": status,
        "expected": round(ca_a, 2),
        "actual": round(sum_t, 2),
        "ecart_pct": round(pct, 2),
        "message": msg,
    }


def check_ca_regions_vs_total(cur) -> Dict:
    """Somme des CA par region (sur ventes_detail) doit egaler le CA total."""
    cur.execute(
        """
        SELECT region, COALESCE(SUM(sales), 0) AS ca_region
        FROM ventes_detail
        GROUP BY region;
        """
    )
    rows = cur.fetchall()
    sum_regions = sum(float(r.get("ca_region") or 0) for r in rows)

    cur.execute("SELECT COALESCE(SUM(sales), 0) AS s FROM ventes_detail;")
    ca_total = float(cur.fetchone()["s"])

    return _check(
        "Somme(CA regions) = CA total ventes_detail",
        expected=ca_total,
        actual=sum_regions,
        message_ok="Somme des CA par region coherente avec le total.",
        message_ko="Somme des CA par region differente du total.",
    )


def check_panier_moyen(cur) -> Dict:
    """Panier moyen doit egaler CA total / nb_commandes."""
    cur.execute(
        """
        SELECT ca_total, nb_commandes, panier_moyen
        FROM kpi_global ORDER BY created_at DESC LIMIT 1;
        """
    )
    row = cur.fetchone()
    if not row:
        return {
            "name": "Panier moyen = CA / Nb_Commandes",
            "status": "warning",
            "expected": None, "actual": None, "ecart_pct": None,
            "message": "Aucune donnee kpi_global.",
        }

    ca = float(row.get("ca_total") or 0)
    nb = int(row.get("nb_commandes") or 0)
    panier = float(row.get("panier_moyen") or 0)
    attendu = (ca / nb) if nb > 0 else None

    return _check(
        "Panier moyen = CA / Nb_Commandes",
        expected=attendu,
        actual=panier,
        message_ok=f"Panier moyen coherent : {panier:.2f} = {ca:,.0f} / {nb:,}.",
        message_ko=f"Panier moyen incoherent : {panier:.2f}, attendu {attendu:.2f}." if attendu else "Donnees insuffisantes.",
    )


def check_no_negative_sales(cur) -> Dict:
    """Aucune valeur de ventes ne doit etre negative dans ventes_detail."""
    cur.execute("SELECT COUNT(*) AS n FROM ventes_detail WHERE sales < 0;")
    nb_neg = int(cur.fetchone()["n"])
    if nb_neg == 0:
        return {
            "name": "Aucune vente negative dans ventes_detail",
            "status": "ok",
            "expected": 0, "actual": 0, "ecart_pct": 0,
            "message": "Aucune ligne avec sales < 0.",
        }
    return {
        "name": "Aucune vente negative dans ventes_detail",
        "status": "error",
        "expected": 0, "actual": nb_neg, "ecart_pct": None,
        "message": f"{nb_neg} ligne(s) avec sales negatif - donnee invalide.",
    }


def check_no_null_dimensions(cur) -> Dict:
    """Aucune ligne ventes_detail ne doit avoir region/category/order_date null."""
    cur.execute(
        """
        SELECT COUNT(*) AS n FROM ventes_detail
        WHERE region IS NULL OR category IS NULL OR order_date IS NULL;
        """
    )
    nb_null = int(cur.fetchone()["n"])
    if nb_null == 0:
        return {
            "name": "Aucune dimension critique NULL dans ventes_detail",
            "status": "ok",
            "expected": 0, "actual": 0, "ecart_pct": 0,
            "message": "Toutes les lignes ont region, category, order_date renseignes.",
        }
    return {
        "name": "Aucune dimension critique NULL dans ventes_detail",
        "status": "warning",
        "expected": 0, "actual": nb_null, "ecart_pct": None,
        "message": f"{nb_null} ligne(s) avec region/category/order_date a NULL.",
    }


def check_croissance_yoy_coherente(cur) -> Dict:
    """
    Verifie que la croissance YoY stockee correspond au calcul reel
    (CA_N - CA_N-1) / CA_N-1 * 100, sur 1% pres.
    """
    cur.execute(
        """
        SELECT annee, ca_annuel, croissance_yoy
        FROM kpi_annuel ORDER BY annee;
        """
    )
    rows = cur.fetchall()
    if len(rows) < 2:
        return {
            "name": "Croissance YoY coherente",
            "status": "warning",
            "expected": None, "actual": None, "ecart_pct": None,
            "message": "Moins de 2 annees, YoY non verifiable.",
        }

    incoherences = []
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur_row = rows[i]
        ca_prev = float(prev.get("ca_annuel") or 0)
        ca_cur = float(cur_row.get("ca_annuel") or 0)
        yoy_stored = cur_row.get("croissance_yoy")
        if yoy_stored is None or ca_prev <= 0:
            continue
        yoy_real = ((ca_cur - ca_prev) / ca_prev) * 100
        ecart = abs(float(yoy_stored) - yoy_real)
        if ecart > 1.0:
            incoherences.append((int(cur_row["annee"]), float(yoy_stored), yoy_real, ecart))

    if not incoherences:
        return {
            "name": "Croissance YoY coherente",
            "status": "ok",
            "expected": None, "actual": None, "ecart_pct": 0,
            "message": "Toutes les croissances YoY sont coherentes avec le calcul brut.",
        }

    annee, stored, real, ecart = max(incoherences, key=lambda x: x[3])
    return {
        "name": "Croissance YoY coherente",
        "status": "warning",
        "expected": round(real, 2),
        "actual": round(stored, 2),
        "ecart_pct": round(ecart, 2),
        "message": f"Ecart {ecart:.2f}pp en {annee} : stockee={stored:.1f}%, reelle={real:.1f}%.",
    }


def run_all_checks() -> Dict:
    """
    Lance tous les checks. Renvoie :
        {
            "status_global": "ok" | "warning" | "error",
            "nb_checks": int,
            "nb_ok": int, "nb_warning": int, "nb_error": int,
            "checks": [ {...}, ... ],
        }
    """
    checks = []
    try:
        with get_cursor() as cur:
            checks.append(check_ca_global_vs_annuel(cur))
            checks.append(check_ca_annuel_vs_trimestriel(cur))
            checks.append(check_ca_regions_vs_total(cur))
            checks.append(check_panier_moyen(cur))
            checks.append(check_no_negative_sales(cur))
            checks.append(check_no_null_dimensions(cur))
            checks.append(check_croissance_yoy_coherente(cur))
    except Exception as e:
        logger.exception(f"Erreur validation business: {e}")
        return {
            "status_global": "error",
            "nb_checks": 0, "nb_ok": 0, "nb_warning": 0, "nb_error": 1,
            "checks": [{
                "name": "Connexion PostgreSQL",
                "status": "error",
                "expected": None, "actual": None, "ecart_pct": None,
                "message": f"Erreur d'acces a la BDD : {e}",
            }],
        }

    nb_ok = sum(1 for c in checks if c["status"] == "ok")
    nb_warn = sum(1 for c in checks if c["status"] == "warning")
    nb_err = sum(1 for c in checks if c["status"] == "error")

    if nb_err > 0:
        status_global = "error"
    elif nb_warn > 0:
        status_global = "warning"
    else:
        status_global = "ok"

    return {
        "status_global": status_global,
        "nb_checks": len(checks),
        "nb_ok": nb_ok,
        "nb_warning": nb_warn,
        "nb_error": nb_err,
        "checks": checks,
    }


if __name__ == "__main__":
    import json
    result = run_all_checks()
    print(json.dumps(result, indent=2, ensure_ascii=False))
