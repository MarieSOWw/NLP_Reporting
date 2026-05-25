"""
embeddings_local.py - Embeddings locaux via sentence-transformers
==================================================================

Role
----
Wrapper haut-niveau autour de sentence-transformers. Permet d'encoder
du texte en vecteurs (embeddings) en local, sans appel API externe.

Pourquoi avoir un module local en plus de Mistral
-------------------------------------------------
- Cout API zero, latence reseau zero.
- Modele compact (all-MiniLM-L6-v2, environ 90 MB, 384 dimensions) qui
  tourne meme sur CPU.
- Le projet exige une demonstration d'un vrai Transformer local, pas
  seulement un appel API.
- L'indexation RAG (load_rag_chunks.py) ET la requete (api.py) doivent
  utiliser le MEME backend pour que les dimensions correspondent dans
  ChromaDB. Le defaut sentence-transformers garantit cette coherence
  (384 dimensions des deux cotes).

Strategie
---------
1. Backend principal : sentence-transformers/all-MiniLM-L6-v2 (384d).
2. Cache singleton pour eviter de recharger le modele a chaque appel.
3. Mistral reste disponible en fallback explicite ailleurs dans le
   projet (api.py:_embed_question, load_rag_chunks.py) mais n'est PAS
   gere ici.

Usage
-----
    from src.embeddings_local import LocalEmbedder
    embedder = LocalEmbedder()
    vec = embedder.embed("Quelle est la meilleure region ?")
    # len(vec) == 384
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model_cache: dict = {}


class LocalEmbedder:
    """Wrapper sentence-transformers avec singleton."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or DEFAULT_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        global _model_cache
        if self.model_name in _model_cache:
            logger.info(f"Modele {self.model_name} charge depuis cache")
            self.model = _model_cache[self.model_name]
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Telechargement modele {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            _model_cache[self.model_name] = self.model
            logger.info(f"Modele {self.model_name} charge")
        except ImportError:
            logger.error(
                "sentence-transformers non installe. "
                "Lancez : pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Erreur chargement modele : {e}")
            raise

    def embed(self, text: str) -> Optional[List[float]]:
        """Encode un texte unique. Renvoie un vecteur de 384 floats."""
        if not self.model:
            logger.warning("Modele non charge")
            return None
        try:
            vec = self.model.encode(text, convert_to_numpy=True)
            return vec.tolist()
        except Exception as e:
            logger.error(f"Erreur embedding : {e}")
            return None

    def embed_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Encode une liste de textes (passage en batch optimise)."""
        if not self.model:
            logger.warning("Modele non charge")
            return None
        try:
            vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [v.tolist() for v in vecs]
        except Exception as e:
            logger.error(f"Erreur embedding batch : {e}")
            return None


def get_embedding_local(text: str, model_name: str = None) -> Optional[List[float]]:
    """Helper one-shot pour integration externe (api.py)."""
    embedder = LocalEmbedder(model_name=model_name)
    return embedder.embed(text)


if __name__ == "__main__":
    import time
    print("\n" + "=" * 70)
    print("   TEST - Embeddings locaux (sentence-transformers)")
    print("=" * 70)

    print("\n[1] Embedding simple")
    embedder = LocalEmbedder()
    text = "Quelle est la meilleure region en termes de ventes ?"
    t0 = time.time()
    vec = embedder.embed(text)
    t1 = time.time()
    print(f"   Texte : {text}")
    print(f"   Dim   : {len(vec)} dimensions")
    print(f"   Duree : {(t1 - t0) * 1000:.1f}ms")

    print("\n[2] Batch")
    texts = [
        "Performance T4 2017",
        "Top 5 produits par categorie",
        "Anomalies detectees",
        "Croissance annuelle 2018",
    ]
    t0 = time.time()
    vecs = embedder.embed_batch(texts)
    t1 = time.time()
    print(f"   Textes : {len(texts)}")
    print(f"   Dim    : {len(vecs[0])}")
    print(f"   Duree  : {(t1 - t0) * 1000:.1f}ms")

    print("\n[3] Similarite cosinus")
    import numpy as np

    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    q1 = "Meilleures ventes"
    q2 = "Top produits"
    q3 = "Anomalies detectees"
    e1, e2, e3 = embedder.embed(q1), embedder.embed(q2), embedder.embed(q3)
    print(f"   '{q1}' vs '{q2}' : {cos_sim(e1, e2):.3f} (similaires)")
    print(f"   '{q1}' vs '{q3}' : {cos_sim(e1, e3):.3f} (differents)")

    print("\n" + "=" * 70)
    print("   [OK] TESTS REUSSIS")
    print("=" * 70 + "\n")
