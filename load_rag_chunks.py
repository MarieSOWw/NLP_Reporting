"""
load_rag_chunks.py - Indexation RAG dans ChromaDB
====================================================

Role
----
Construit les chunks de texte qui alimentent le chatbot RAG, calcule
leurs embeddings, et les insere dans ChromaDB (vector store local
persistant dans `.chroma_rag/`). Sert de base de connaissances que
api.py:/api/chat interroge a chaque question utilisateur.

Pourquoi sentence-transformers cote indexation ET cote requete
--------------------------------------------------------------
Les embeddings stockes dans ChromaDB doivent imperativement avoir la
meme dimension que ceux generes au moment de la requete par
api.py:_embed_question. Si on indexait en Mistral (1024d) puis on
requetait en sentence-transformers (384d), ChromaDB leverait une
exception capturee silencieusement et `search_rag_chunks` retournerait
toujours liste vide.

Le backend par defaut est donc sentence-transformers/all-MiniLM-L6-v2
(384 dimensions) des deux cotes. Mistral n'est utilise que comme
fallback explicite si sentence-transformers est absent (avec
avertissement fort dans les logs).

Sources des chunks (extraits via PostgreSQL)
--------------------------------------------
- kpi_global, kpi_annuel : metriques business
- top_products (sub_category top 10 via ventes_detail)
- anomalies : YoY > 20%
- segments, regions : breakdown via ventes_detail
- rapports_nlp : decoupage en chunks de 250 mots, plus un chunk
  resume_bullets

Usage
-----
    python load_rag_chunks.py
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
import chromadb

try:
    from config import (
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
        MISTRAL_API_KEY, BASE_DIR,
    )
except ImportError:
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "nlp_reporting"
    DB_USER = "postgres"
    DB_PASSWORD = "admin123"
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    BASE_DIR = PROJECT_ROOT

CHROMA_PATH = str(BASE_DIR / ".chroma_rag")
CHROMA_COLLECTION = "rag_chunks"

SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBED_DIM = 384

MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
MISTRAL_EMBED_DIM = 1024

BATCH_SIZE = 32
RATE_LIMIT_SLEEP = 0.5

_st_model = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def get_chroma_collection():
    """Recree la collection ChromaDB persistante (reset complet)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(CHROMA_COLLECTION)
        logger.info("  Collection existante supprimee")
    except Exception:
        pass
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def chunk_kpis_global(conn) -> List[dict]:
    chunks = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ca_total, nb_commandes, panier_moyen, nb_clients, nb_produits,
                   croissance_globale, periode, meilleure_annee, meilleur_ca,
                   meilleure_region
            FROM kpi_global ORDER BY created_at DESC LIMIT 1;
            """
        )
        row = cur.fetchone()
        if not row:
            return chunks
        ca, nb_cmd, panier, clients, produits, cr, periode, m_an, m_ca, m_reg = row
        chunks.append({
            "chunk_type": "kpi",
            "report_type": "global",
            "filter_year": None,
            "filter_region": None,
            "filter_category": None,
            "content": (
                f"Sur la periode {periode}, le chiffre d'affaires total Superstore est de "
                f"{ca:,.0f} USD, genere par {nb_cmd:,} commandes uniques avec un panier moyen "
                f"de {panier:.2f} USD. Le portefeuille client compte {clients:,} clients actifs "
                f"sur {produits:,} produits referencees. La croissance globale sur la periode est "
                f"de {cr}%. La meilleure annee est {m_an} avec {m_ca:,.0f} USD de revenus, "
                f"et la region la plus performante est {m_reg}."
            ),
            "metadata": {
                "ca_total": float(ca), "nb_commandes": int(nb_cmd),
                "panier_moyen": float(panier), "nb_clients": int(clients),
                "croissance": float(cr), "meilleure_region": m_reg,
                "meilleure_annee": int(m_an) if m_an else None,
            },
        })
    return chunks


def chunk_kpis_annual(conn) -> List[dict]:
    chunks = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT annee, ca_annuel, nb_commandes, panier_moyen,
                   clients_uniques, croissance_yoy
            FROM kpi_annuel ORDER BY annee;
            """
        )
        for row in cur.fetchall():
            an, ca, nb_cmd, panier, clients, yoy = row
            yoy_str = f"croissance YoY de {yoy}%" if yoy else "premiere annee (pas de YoY)"
            chunks.append({
                "chunk_type": "kpi",
                "report_type": "by_year",
                "filter_year": int(an),
                "filter_region": None,
                "filter_category": None,
                "content": (
                    f"En {an}, le chiffre d'affaires annuel atteint {ca:,.0f} USD avec "
                    f"{nb_cmd:,} commandes (panier moyen : {panier:.2f} USD), "
                    f"servant {clients:,} clients uniques. {yoy_str.capitalize()}."
                ),
                "metadata": {
                    "annee": int(an), "ca_annuel": float(ca),
                    "nb_commandes": int(nb_cmd), "panier_moyen": float(panier),
                    "croissance_yoy": float(yoy) if yoy else None,
                },
            })
    return chunks


def chunk_top_products(conn) -> List[dict]:
    chunks = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sub_category,
                   SUM(sales) as ca,
                   COUNT(DISTINCT order_id) as nb_cmd,
                   COUNT(DISTINCT customer_id) as clients
            FROM ventes_detail
            GROUP BY sub_category
            ORDER BY ca DESC
            LIMIT 10;
            """
        )
        for rank, (sub, ca, nb_cmd, clients) in enumerate(cur.fetchall(), 1):
            chunks.append({
                "chunk_type": "top_product",
                "report_type": "global",
                "filter_year": None,
                "filter_region": None,
                "filter_category": None,
                "content": (
                    f"La sous-categorie {sub} occupe le rang {rank} du top vente avec "
                    f"{ca:,.0f} USD de chiffre d'affaires sur {nb_cmd:,} commandes, "
                    f"servant {clients:,} clients differents."
                ),
                "metadata": {
                    "sub_category": sub, "rank": rank,
                    "ca": float(ca), "nb_commandes": int(nb_cmd),
                },
            })
    return chunks


def chunk_anomalies(conn) -> List[dict]:
    chunks = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT niveau, type_anomalie, annee, trimestre, variation, ventes, description
            FROM anomalies
            ORDER BY ABS(variation) DESC;
            """
        )
        for niveau, typ, an, trim, var, ventes, desc in cur.fetchall():
            chunks.append({
                "chunk_type": "anomalie",
                "report_type": "global",
                "filter_year": int(an),
                "filter_region": None,
                "filter_category": None,
                "content": (
                    f"Au T{trim} {an}, une anomalie de niveau {niveau} a ete detectee : "
                    f"variation de {var}% par rapport au trimestre precedent "
                    f"(ventes : {ventes:,.0f} USD). {desc}"
                ),
                "metadata": {
                    "niveau": niveau, "annee": int(an), "trimestre": int(trim),
                    "variation": float(var), "ventes": float(ventes),
                },
            })
    return chunks


def chunk_segments(conn) -> List[dict]:
    chunks = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT segment,
                   SUM(sales) as ca,
                   COUNT(DISTINCT order_id) as nb_cmd,
                   COUNT(DISTINCT customer_id) as clients
            FROM ventes_detail
            GROUP BY segment
            ORDER BY ca DESC;
            """
        )
        rows = cur.fetchall()
        total = sum(r[1] for r in rows) or 1
        for seg, ca, nb_cmd, clients in rows:
            part = (ca / total) * 100
            chunks.append({
                "chunk_type": "segment",
                "report_type": "global",
                "filter_year": None,
                "filter_region": None,
                "filter_category": None,
                "content": (
                    f"Le segment client {seg} represente {part:.1f}% du chiffre d'affaires "
                    f"total avec {ca:,.0f} USD generes sur {nb_cmd:,} commandes "
                    f"par {clients:,} clients uniques."
                ),
                "metadata": {
                    "segment": seg, "ca": float(ca),
                    "part_pct": round(part, 1), "nb_clients": int(clients),
                },
            })
    return chunks


def chunk_regions(conn) -> List[dict]:
    chunks = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT region,
                   SUM(sales) as ca,
                   COUNT(DISTINCT order_id) as nb_cmd,
                   COUNT(DISTINCT customer_id) as clients
            FROM ventes_detail
            GROUP BY region
            ORDER BY ca DESC;
            """
        )
        rows = cur.fetchall()
        total = sum(r[1] for r in rows) or 1
        for reg, ca, nb_cmd, clients in rows:
            part = (ca / total) * 100
            chunks.append({
                "chunk_type": "region",
                "report_type": "by_region",
                "filter_year": None,
                "filter_region": reg,
                "filter_category": None,
                "content": (
                    f"La region {reg} genere {ca:,.0f} USD de chiffre d'affaires "
                    f"({part:.1f}% du total) sur {nb_cmd:,} commandes aupres de "
                    f"{clients:,} clients uniques."
                ),
                "metadata": {
                    "region": reg, "ca": float(ca), "part_pct": round(part, 1),
                },
            })
    return chunks


def chunk_nlp_reports(conn) -> List[dict]:
    chunks = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, langue, report_type, filter_year, filter_region,
                   filter_category, rapport_complet,
                   resume_bullet1, resume_bullet2, resume_bullet3
            FROM rapports_nlp
            WHERE langue = 'fr'
            ORDER BY generated_at DESC;
            """
        )
        for row in cur.fetchall():
            (rid, lang, rtype, fy, fr, fc, rapport, b1, b2, b3) = row
            if rapport:
                paragraphs = [p.strip() for p in rapport.split("\n") if p.strip()]
                current, current_words = [], 0
                for para in paragraphs:
                    nb = len(para.split())
                    if current_words + nb > 250 and current:
                        chunks.append(_make_nlp_chunk("\n".join(current), rtype, fy, fr, fc))
                        current, current_words = [], 0
                    current.append(para)
                    current_words += nb
                if current:
                    chunks.append(_make_nlp_chunk("\n".join(current), rtype, fy, fr, fc))
            bullets = [b for b in [b1, b2, b3] if b]
            if bullets:
                chunks.append({
                    "chunk_type": "rapport_nlp",
                    "report_type": rtype,
                    "filter_year": fy,
                    "filter_region": fr,
                    "filter_category": fc,
                    "content": "Resume executif :\n" + "\n".join(f"- {b}" for b in bullets),
                    "metadata": {"bullets_count": len(bullets), "report_id": int(rid)},
                })
    return chunks


def _make_nlp_chunk(content, rtype, fy, fr, fc):
    return {
        "chunk_type": "rapport_nlp",
        "report_type": rtype,
        "filter_year": fy,
        "filter_region": fr,
        "filter_category": fc,
        "content": content,
        "metadata": {"word_count": len(content.split())},
    }


def _get_sentence_transformer():
    """Charge sentence-transformers une seule fois (singleton)."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Chargement {SENTENCE_TRANSFORMER_MODEL}...")
            _st_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            logger.info(f"Sentence-Transformer charge (dim={EMBED_DIM})")
        except Exception as e:
            logger.warning(f"sentence-transformers indisponible: {e}")
            _st_model = False
    return _st_model if _st_model is not False else None


def _embed_with_sentence_transformers(texts: List[str]) -> Optional[List[List[float]]]:
    model = _get_sentence_transformer()
    if model is None:
        return None

    logger.info(f"  Encodage local de {len(texts)} chunks...")
    embeddings = []
    nb_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1
        logger.info(f"  Batch {batch_idx}/{nb_batches} ({len(batch)} chunks)")
        try:
            vecs = model.encode(batch, convert_to_numpy=False, show_progress_bar=False)
            for v in vecs:
                if hasattr(v, "tolist"):
                    embeddings.append(v.tolist())
                else:
                    embeddings.append(list(v))
        except Exception as e:
            logger.error(f"  Erreur encodage batch {batch_idx}: {e}")
            embeddings.extend([[0.0] * EMBED_DIM] * len(batch))
    return embeddings


def _embed_with_mistral(texts: List[str]) -> List[List[float]]:
    """Fallback Mistral. ATTENTION : 1024d, incompatible avec sentence-transformers."""
    logger.warning(
        "FALLBACK MISTRAL ACTIVE (1024d). "
        "La recherche RAG ne fonctionnera que si api.py:_embed_question utilise "
        "aussi Mistral. Sinon mismatch dimension."
    )
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY manquante - fallback impossible")
        return [[0.0] * MISTRAL_EMBED_DIM] * len(texts)
    try:
        from mistralai.client import Mistral
        client = Mistral(api_key=MISTRAL_API_KEY)
    except ImportError:
        logger.error("SDK mistralai non installe (pip install mistralai)")
        return [[0.0] * MISTRAL_EMBED_DIM] * len(texts)

    all_embeddings = []
    nb_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1
        logger.info(f"  Mistral batch {batch_idx}/{nb_batches} ({len(batch)} chunks)")
        try:
            resp = client.embeddings.create(model=MISTRAL_EMBED_MODEL, inputs=batch)
            all_embeddings.extend([item.embedding for item in resp.data])
            time.sleep(RATE_LIMIT_SLEEP)
        except Exception as e:
            logger.error(f"  Erreur batch Mistral {batch_idx}: {e}")
            all_embeddings.extend([[0.0] * MISTRAL_EMBED_DIM] * len(batch))
    return all_embeddings


def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = _embed_with_sentence_transformers(texts)
    if embeddings is not None and len(embeddings) == len(texts):
        return embeddings
    logger.warning("sentence-transformers indisponible - bascule Mistral")
    return _embed_with_mistral(texts)


def insert_chunks_chroma(chunks: List[dict], embeddings: List[List[float]]):
    """Insere chunks + embeddings dans ChromaDB par batches de 100."""
    collection = get_chroma_collection()
    ids, documents, metadatas, embs = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        ids.append(str(i))
        documents.append(chunk["content"])
        meta = {
            "chunk_type": chunk["chunk_type"] or "",
            "report_type": chunk["report_type"] or "",
            "filter_year": chunk["filter_year"] if chunk["filter_year"] is not None else -1,
            "filter_region": chunk["filter_region"] or "",
            "filter_category": chunk["filter_category"] or "",
        }
        for k, v in (chunk.get("metadata") or {}).items():
            if isinstance(v, (str, int, float, bool)) and v is not None:
                meta[k] = v
        metadatas.append(meta)
        embs.append(emb)

    CHROMA_BATCH = 100
    for start in range(0, len(ids), CHROMA_BATCH):
        end = start + CHROMA_BATCH
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embs[start:end],
        )

    logger.info(f"[OK] {len(chunks)} chunks indexes dans ChromaDB ({CHROMA_PATH})")
    return collection


def main():
    logger.info("Indexation RAG - ChromaDB + Sentence-Transformers")
    logger.info(f"   Stockage ChromaDB : {CHROMA_PATH}")
    logger.info(f"   Modele embeddings : {SENTENCE_TRANSFORMER_MODEL} ({EMBED_DIM}d)")

    has_st = _get_sentence_transformer() is not None
    has_mistral = bool(MISTRAL_API_KEY)
    if not (has_st or has_mistral):
        logger.error(
            "Aucun backend d'embeddings disponible. Installe "
            "sentence-transformers (recommande) OU configure MISTRAL_API_KEY."
        )
        sys.exit(1)
    if not has_st:
        logger.warning(
            "sentence-transformers indisponible. "
            "Indexation Mistral (1024d) - api.py doit alors aussi utiliser Mistral."
        )

    conn = get_conn()
    try:
        logger.info("Construction des chunks depuis PostgreSQL...")
        chunks = []
        chunks += chunk_kpis_global(conn)
        chunks += chunk_kpis_annual(conn)
        chunks += chunk_top_products(conn)
        chunks += chunk_anomalies(conn)
        chunks += chunk_segments(conn)
        chunks += chunk_regions(conn)
        chunks += chunk_nlp_reports(conn)

        if not chunks:
            logger.error("Aucun chunk genere (PostgreSQL vide ?)")
            sys.exit(1)

        logger.info(f"[OK] {len(chunks)} chunks construits :")
        from collections import Counter
        for chunk_type, count in Counter(c["chunk_type"] for c in chunks).items():
            logger.info(f"   {chunk_type:15s} : {count}")

        logger.info("Generation des embeddings...")
        texts = [c["content"] for c in chunks]
        embeddings = embed_texts(texts)

        if embeddings:
            actual_dim = len(embeddings[0])
            logger.info(f"   Dimension obtenue : {actual_dim}")
            if actual_dim not in (EMBED_DIM, MISTRAL_EMBED_DIM):
                logger.warning(
                    f"   Dimension inattendue {actual_dim} "
                    f"(attendue {EMBED_DIM} ou {MISTRAL_EMBED_DIM})"
                )

        logger.info("Insertion dans ChromaDB...")
        collection = insert_chunks_chroma(chunks, embeddings)

        total = collection.count()
        logger.info(f"\n[OK] Indexation terminee : {total} chunks dans ChromaDB")
        logger.info("Le chatbot RAG est maintenant operationnel.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
