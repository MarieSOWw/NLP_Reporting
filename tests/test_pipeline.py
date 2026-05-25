"""
test_pipeline.py - Tests unitaires du pipeline NLP Reporting
==============================================================

Role
----
Tests pytest standards. Couvre 4 zones :
    - Configuration  (import du module config.py, existence des chemins)
    - NLTKProcessor  (tokenisation, sentiment, extraction de mots-cles)
    - Score business (calculer_score sur faits synthetiques)
    - Anomalies      (detecter_anomalies YoY)
    - Resume executif

Difference avec tests/test_complet.py
--------------------------------------
test_pipeline.py = tests unitaires pytest, sans DB ni API reelle,
sur composants isoles.
test_complet.py = smoke test manuel, lance sur l'environnement reel.

Usage
-----
    pytest tests/test_pipeline.py -v
"""

import pytest
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfig:
    """Tests du module de configuration."""
    
    def test_config_import(self):
        """Test que le module config s'importe correctement."""
        from config import BASE_DIR, DATA_PATH, OUTPUT_DIR
        assert BASE_DIR is not None
        assert DATA_PATH is not None
        assert OUTPUT_DIR is not None
    
    def test_config_paths_exist(self):
        """Test que les répertoires existent."""
        from config import BASE_DIR, OUTPUT_DIR
        assert BASE_DIR.exists()
        # OUTPUT_DIR est créé automatiquement


class TestNLTK:
    """Tests du module NLTK."""
    
    def test_nltk_processor_init(self):
        """Test l'initialisation du processeur NLTK."""
        from src.nlp_nltk import NLTKProcessor
        processor = NLTKProcessor(langue="english")
        assert processor is not None
        assert processor.langue == "english"
    
    def test_tokenize_sentences(self):
        """Test la tokenization des phrases."""
        from src.nlp_nltk import NLTKProcessor
        processor = NLTKProcessor()
        text = "Hello world. This is a test."
        sentences = processor.tokenize_sentences(text)
        assert len(sentences) >= 2
    
    def test_tokenize_words(self):
        """Test la tokenization des mots."""
        from src.nlp_nltk import NLTKProcessor
        processor = NLTKProcessor()
        text = "Hello world"
        words = processor.tokenize_words(text)
        assert len(words) == 2
    
    def test_sentiment_analysis(self):
        """Test l'analyse de sentiment."""
        from src.nlp_nltk import NLTKProcessor
        processor = NLTKProcessor()
        
        positive_text = "This is excellent and wonderful!"
        sentiment = processor.analyze_sentiment(positive_text)
        assert sentiment['compound'] > 0
        
        negative_text = "This is terrible and awful!"
        sentiment = processor.analyze_sentiment(negative_text)
        assert sentiment['compound'] < 0
    
    def test_extract_keywords(self):
        """Test l'extraction de mots-clés.

        FIX TEST -Avant : on attendait 'sale' ou 'revenue' dans le top.
        Mais `NLTKProcessor.__init__` ajoute explicitement `sales`, `revenue`,
        `growth`, etc. à ses stopwords personnalisés (cf. nlp_nltk.py:332-335 :
        choix design pour éviter ces termes business génériques dans les
        keywords). Le test doit donc utiliser des mots qui NE sont PAS
        stopwords, comme 'performance' ou 'meeting'.
        """
        from src.nlp_nltk import NLTKProcessor
        processor = NLTKProcessor()
        text = "Performance meeting performance dashboard meeting dashboard analytics"
        keywords = processor.extract_keywords(text, top_n=3)
        assert len(keywords) <= 3
        assert len(keywords) >= 1, "Au moins un keyword devrait être extrait"
        keyword_words = [k[0] for k in keywords]
        # 'performance', 'meeting' ou 'dashboard' (lemmatisés) doivent être présents
        attendus = ['performance', 'meeting', 'dashboard', 'analytics']
        assert any(w in keyword_words for w in attendus), (
            f"Aucun mot attendu trouvé. Keywords obtenus : {keyword_words}"
        )


class TestScore:
    """Tests du calcul de score."""
    
    def test_score_calculation(self):
        """Test le calcul du score de performance."""
        from src.nlp_transformers import calculer_score
        
        # Créer des faits de test
        faits = {
            'croissance_globale': 50.0,
            'variations': [
                {'variation': 10.0},
                {'variation': -5.0},
                {'variation': 15.0},
            ],
            'meilleur_trim': {'variation': 30.0},
            'top_produits': [
                {'ventes': 100000},
                {'ventes': 80000},
                {'ventes': 60000},
            ],
        }
        
        score = calculer_score(faits)
        
        assert 'score' in score
        assert 'mention' in score
        assert 'details' in score
        assert 0 <= score['score'] <= 100
    
    def test_score_mention(self):
        """Test que la mention correspond au score."""
        from src.nlp_transformers import calculer_score
        
        faits_excellent = {
            'croissance_globale': 100.0,
            'variations': [{'variation': 5.0}],
            'meilleur_trim': {'variation': 50.0},
            'top_produits': [
                {'ventes': 100},
                {'ventes': 100},
                {'ventes': 100},
            ],
        }
        
        score = calculer_score(faits_excellent)
        assert score['score'] >= 70


class TestAnomalies:
    """Tests de la détection d'anomalies."""
    
    def test_detect_anomalies(self):
        """Test la détection d'anomalies (YoY).

        FIX TEST -Avant : on passait 3 trimestres d'UNE seule année
        avec variation pré-remplie, en attendant 2 anomalies. Mais depuis
        la v5.1, `detecter_anomalies` fonctionne en YoY (compare T_i
        année N à T_i année N-1) pour éliminer les faux positifs
        saisonniers. Sur une seule année, aucune anomalie YoY n'est
        calculable → résultat correct = 0. Nouveau jeu de test : on
        couvre 2 années sur les mêmes trimestres pour permettre la
        comparaison YoY.
        """
        from src.nlp_transformers import detecter_anomalies

        # T1 : -40% YoY (alerte), T2 : +50% YoY (info), T3 : ~+10% (rien)
        faits = {
            'variations': [
                {'annee': 2019, 'trimestre': 1, 'variation': 0.0, 'ventes': 100000},
                {'annee': 2020, 'trimestre': 1, 'variation': 0.0, 'ventes': 60000},
                {'annee': 2019, 'trimestre': 2, 'variation': 0.0, 'ventes': 80000},
                {'annee': 2020, 'trimestre': 2, 'variation': 0.0, 'ventes': 120000},
                {'annee': 2019, 'trimestre': 3, 'variation': 0.0, 'ventes': 100000},
                {'annee': 2020, 'trimestre': 3, 'variation': 0.0, 'ventes': 110000},
            ]
        }

        anomalies = detecter_anomalies(faits, seuil_alerte=20.0)

        # T1 et T2 doivent ressortir (|var| ≥ 20%), pas T3
        assert len(anomalies) == 2, f"2 anomalies attendues, obtenu {len(anomalies)}"

        # Vérification structurelle : on a au moins une ALERTE et son type
        alertes = [a for a in anomalies if a['niveau'] == 'ALERTE']
        infos = [a for a in anomalies if a['niveau'] == 'INFO']
        assert len(alertes) == 1 and len(infos) == 1
        assert alertes[0]['type'] == 'chute_yoy'
        assert alertes[0]['variation'] < 0


class TestResumeExecutif:
    """Tests du résumé exécutif."""
    
    def test_resume_generation(self):
        """Test la génération du résumé exécutif."""
        from src.nlp_transformers import generer_resume_executif
        
        faits = {
            'meilleure_annee': {'annee': 2020, 'ca': 1000000, 'commandes': 5000},
            'meilleur_trim': {'annee': 2020, 'trimestre': 4, 'variation': 25.0},
            'top_produits': [{'nom': 'Phones', 'ventes': 300000}],
            'croissance_globale': 50.0,
            'periode': '2018-2020',
            'pire_trim': {'annee': 2019, 'trimestre': 1, 'variation': -10.0},
            'variations': [],
        }
        
        resume_fr = generer_resume_executif(faits, langue='fr')
        resume_en = generer_resume_executif(faits, langue='en')
        
        assert 'titre' in resume_fr
        assert 'bullets' in resume_fr
        assert len(resume_fr['bullets']) == 3
        
        # Accepte avec ou sans accents (la convention sobre du projet
        # n'inclut plus les accents dans les titres techniques).
        titre_fr = resume_fr['titre'].upper()
        assert ('RESUME' in titre_fr) or ('RÉSUMÉ' in titre_fr)
        assert 'SUMMARY' in resume_en['titre'].upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
