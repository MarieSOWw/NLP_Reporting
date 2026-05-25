# NLP Business Reporting Platform

> Plateforme end-to-end qui transforme un dataset de ventes brut en rapports business automatises : analyses executives, dashboards interactifs, PDFs livrables et plan d'action hierarchise.

Projet academique ISM. Pipeline `Spark -> NLTK -> Mistral (Transformer) -> PostgreSQL -> FastAPI -> PDF/Dashboard`.

---

## Table des matieres

1. Le probleme adresse
2. Architecture du pipeline
3. Stack technique et choix de design
4. Structure du projet
5. Les 12 KPIs business
6. Les quatre chantiers NLP
7. Schema PostgreSQL et coherence des donnees
8. Endpoints API et chatbot RAG
9. Installation et execution
10. Tests et validation
11. Decisions architecturales et conventions
12. Limites connues et axes d'evolution

---

## 1. Le probleme adresse

Dans une entreprise, la chaine entre "donnees brutes" et "decision business" est longue, couteuse et bruyante.

1. Un analyste passe des heures a extraire, croiser et agreger des KPIs depuis des CSV.
2. Il met en forme dans Excel, ecrit un commentaire, genere des graphiques.
3. Un manager lit le document, oublie la moitie des chiffres, et demande une version filtree par region ou par annee.
4. On recommence.

Ce projet industrialise entierement cette chaine : dataset en entree -> rapport executif pret a signer en sortie, avec anomalies detectees, score qualite, et recommandations hierarchisees par priorite, niveau et confiance.

---

## 2. Architecture du pipeline

```
Data (CSV - 9 800 lignes Superstore 2015-2018)
   |
   v
+-------------------------------------+
|  Apache Spark                       |  Chargement, nettoyage, 12 KPIs
+-------------------------------------+
   |
   v
+-------------------------------------+
|  NLTK - Extraction de faits         |  Tokenisation, NER, faits structures
+-------------------------------------+
   |
   v
+-------------------------------------+
|  Mistral (Transformer) - 2 etapes   |  Chantier 1 : JSON valide -> prose
|  1. generer_structure_json()        |
|  2. _generer_texte_depuis_structure |
+-------------------------------------+
   |
   v
+-------------------------------------+
|  NLTK - Analyse du rapport genere   |  Chantier 2 : classification business
|  + Tonalite business (lexique)      |  Remplacement de VADER
+-------------------------------------+
   |
   v
+-------------------------------------+
|  Score qualite NLP (6 criteres)     |  Chantier 3 : note /100
+-------------------------------------+
   |
   v
+-------------------------------------+
|  PostgreSQL (11 tables)             |  Source unique de verite
+-------------------------------------+
   |
   +-->  Grafana (dashboard BI)
   +-->  FastAPI (20+ endpoints)
   +-->  PDF (ReportLab + badges recos Chantier 4)
   +-->  Dashboard HTML + Chatbot RAG
```

Chaque maillon a une responsabilite unique et un fichier dedie. Aucun composant aval ne recalcule un KPI : ils lisent tous depuis PostgreSQL, ce qui evite la derive entre PDF, dashboard et API.

---

## 3. Stack technique et choix de design

| Composant | Technologie | Pourquoi ce choix |
|-----------|-------------|-------------------|
| Traitement donnees | Apache Spark (PySpark) | Pandas suffirait pour 9 800 lignes, mais le code est ecrit pour passer a l'echelle sans reecriture. Window functions natives pour YoY/QoQ |
| NLP extraction et analyse | NLTK | Couvre toute la chaine : tokenisation, NER, POS, classification, sentiment. Valorise la stack NLP classique du programme |
| NLP generation | Mistral API (Transformer) | Architecture Transformer attendue par le cahier des charges. JSON structure puis prose narrative (anti-hallucination) |
| Score qualite NLP | Python custom | Aucune librairie ne fait ca specifiquement. 6 criteres ponderes, mesurables et reproductibles |
| Base de donnees | PostgreSQL | Robuste, gratuit, supporte JSONB pour le stockage de structures. Grafana branche dessus en natif |
| Vector store (RAG) | ChromaDB (local persistant) | Pas besoin d'extension Postgres. Tourne en process Python, persiste sur disque |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 384 dimensions, ~90 MB, tourne sur CPU. Coherent indexation et requete (meme modele des deux cotes) |
| Dashboard BI | Grafana | Connexion native PostgreSQL, filtres interactifs |
| API REST | FastAPI | Swagger auto, validation Pydantic, lifespan moderne |
| PDF | ReportLab + Matplotlib | Pas de dependance Tex/Pandoc. Cartes KPI, graphiques business, badges hierarchises |
| Frontend | HTML/CSS/JS + Chart.js | Simple, sans framework, chargement instantane |
| Tests | pytest | Standard Python |

---

## 4. Structure du projet

```
NLP_REPORTING/
|
+-- config.py                  Configuration centralisee (DB, API, NLP, seuils)
+-- main.py                    Point d'entree du pipeline complet
+-- api.py                     Serveur FastAPI + chatbot RAG
+-- nlp_quality_score.py       Chantier 3 : score qualite NLP /100
+-- load_detail_table.py       Chargement ventes_detail (source filtres)
+-- load_rag_chunks.py         Indexation ChromaDB pour le chatbot
+-- requirements.txt           Dependances Python
+-- grafana.json               Dashboard Grafana (import direct)
+-- README.md                  Ce fichier
+-- .env                       Secrets (non commite)
|
+-- src/
|   +-- load_data.py           [1] Chargement Spark
|   +-- analytics.py           [2] Calcul des 12 KPIs
|   +-- nlp_nltk.py            [3] Extraction NLTK + Chantier 2 + tonalite business
|   +-- nlp_transformers.py    [4] Mistral - Chantier 1 (2 etapes) + Chantier 4
|   +-- save_to_postgres.py    [5] Persistence des KPIs + rapports
|   +-- db.py                  Lectures PostgreSQL filtrees (API, PDF)
|   +-- generate_pdf.py        [6] Export PDF + badges Chantier 4
|   +-- pdf_nltk_section.py    Section NLTK du PDF (tonalite, themes)
|   +-- pdf_design_professional.py   Palette et styles editoriaux
|   +-- pdf_images_online.py   Photos Unsplash + cache local
|   +-- embeddings_local.py    Wrapper sentence-transformers
|   +-- text_utils.py          Utilitaires partages (strip_markdown, etc.)
|   +-- purger_cache_nlp.py    Utilitaire de purge du cache obsolete
|
+-- tests/
|   +-- test_pipeline.py       Tests unitaires pytest
|   +-- test_complet.py        Smoke test manuel (DB, API, modeles reels)
|
+-- dashboard/
|   +-- index.html             Frontend HTML/CSS/JS
|
+-- data/
|   +-- train.csv              Superstore Sales (9 800 lignes)
|
+-- outputs/                   PDFs generes (gitignored)
+-- logs/                      Logs d'execution (gitignored)
+-- .cache/                    NLTK data + sentence-transformers cache
+-- .chroma_rag/               ChromaDB persistant
```

---

## 5. Les 12 KPIs business

Tous calcules par `src/analytics.py:compute_kpis()`. Chaque KPI est un DataFrame Spark separe.

| # | KPI | Technique Spark |
|---|---|---|
| 1 | Ventes par region et trimestre | groupBy + agg |
| 2 | Ventes par categorie et region | groupBy + pivot |
| 3 | Top 10 sous-categories | groupBy + orderBy + limit |
| 4 | Variations trimestrielles QoQ et YoY | Window + lag |
| 5 | Performance annuelle avec YoY | Window + lag |
| 6 | Meilleure region par annee | Window + rank |
| 7 | Ventes mensuelles + moyenne mobile 3M | Window + avg over rows |
| 8 | Ventes par segment client | groupBy |
| 9 | Performance par mode de livraison | groupBy + agg |
| 10 | Top 10 clients (RFM simplifie) | groupBy + agg + rank |
| 11 | Analyse par etat americain | groupBy |
| 12 | Saisonnalite jour de semaine | groupBy + dayofweek |

### Definitions metier canoniques

Identiques partout (Spark, PostgreSQL, API, PDF) :

- `Nb_Commandes` = `countDistinct(Order_ID)`. Compte les VRAIES commandes, pas les lignes du CSV.
- `Nb_Articles` = `count(*)`. Une ligne du CSV = un article d'une commande.
- `Panier_Moyen` = `SUM(Sales) / Nb_Commandes`. Panier moyen PAR COMMANDE.
- `Croissance YoY` = variation entre l'annee N et l'annee N-1.
- `Variation QoQ` = variation entre le trimestre courant et le trimestre precedent.
- `Variation YoY` = variation entre meme trimestre annee precedente (utilisee pour les anomalies).

---

## 6. Les quatre chantiers NLP

### Chantier 1 - Generation en 2 etapes (anti-hallucination)

Fichier : `src/nlp_transformers.py`

Au lieu de demander directement un texte a Mistral (risque de derive numerique), on procede en 2 etapes :

```
faits -> Etape 1 : generer_structure_json()        -> JSON valide
                   (resume, risques, opportunites, recos)
      -> Etape 2 : _generer_texte_depuis_structure -> prose narrative
```

Le LLM ne voit JAMAIS les chiffres bruts a recalculer : il produit d'abord un JSON strict (valide contre les faits), puis le transforme en texte. Si un champ est absent ou incoherent, on le remplace par un fallback base sur les faits reels.

Cela garantit que les chiffres dans le texte correspondent exactement aux KPIs.

### Chantier 2 - Classification d'intentions business

Fichier : `src/nlp_nltk.py` (fonction `classifier_phrases_business`)

Chaque phrase du rapport est tagguee par intention :
- `tendance` : decrit ce qui se passe
- `anomalie` : signale un ecart inhabituel
- `opportunite` : identifie un levier
- `risque` : alerte sur un point de vigilance
- `recommandation` : propose une action
- `contexte` : tout le reste

Permet de mesurer la couverture business du rapport via `mesurer_couverture_business()`. Un bon rapport contient au moins une tendance, une alerte (anomalie ou risque), deux recommandations, une opportunite.

### Chantier 3 - Score qualite NLP /100

Fichier : `nlp_quality_score.py`

Six criteres ponderes :

| Critere | Poids | Ce qu'on verifie |
|---|---|---|
| Couverture des faits | 25 | Le texte cite-t-il la meilleure annee, le pire trimestre, le top produit ? |
| Ancrage numerique | 25 | Combien de chiffres concrets, montants, pourcentages dans le texte ? |
| Presence de recommandations | 15 | Au moins 2 actions concretes ? |
| Clarte / lisibilite | 15 | Longueur moyenne des phrases, ratio de mots uniques |
| Absence de repetition | 10 | Diversite des bigrammes |
| Ton business | 10 | Vocabulaire metier present, pas de mots vagues |

Le score est stocke (`rapports_nlp.score_nlp`), affiche dans la console, le PDF et l'API. Permet de comparer deux runs ou detecter une regression de prompt.

### Chantier 4 - Recommandations hierarchisees

Fichier : `src/nlp_transformers.py` + `src/generate_pdf.py`

Chaque recommandation porte trois attributs :

- **Priorite** : `haute` / `moyenne` / `basse` - ce qu'on fait en premier
- **Niveau** : `strategique` / `tactique` / `operationnel` - qui s'en occupe (CODIR, manager, equipe)
- **Confiance** : `elevee` / `moyenne` / `faible` - derivee automatiquement de la volatilite des donnees
- **Justification** chiffree obligatoire

Les recos sont triees automatiquement par priorite et rendues dans le PDF avec des badges colores.

### Finitions

- Lexique business enrichi (12 themes, ~130 termes bilingues) pour la detection de sujets dominants.
- Remplacement du sentiment VADER (peu fiable en B2B et non francophone) par une tonalite business via lexique metier (favorable / neutre / defavorable + score borne [-100, +100]).
- Endpoints API dedies : `/api/nlp/recommendations`, `/api/nlp/quality`, `/api/nlp/tonalite`.

---

## 7. Schema PostgreSQL et coherence des donnees

### Les 12 tables

| Table | Role | Source |
|-------|------|--------|
| `ventes_detail` | Grain transactionnel (1 ligne = 1 article) | load_detail_table.py |
| `kpi_global` | Recap global (CA total, croissance, top region) | save_to_postgres.py |
| `kpi_annuel` | Performance par annee + YoY | save_to_postgres.py |
| `kpi_region_trim` | Ventes region x trimestre | save_to_postgres.py |
| `kpi_variation` | Variations QoQ | save_to_postgres.py |
| `kpi_top_produits` | Top 10 sous-categories | save_to_postgres.py |
| `kpi_mensuel` | Ventes par mois + moyenne mobile | save_to_postgres.py |
| `kpi_segment` | Ventes par segment client | save_to_postgres.py |
| `kpi_categorie` | Ventes categorie x region x annee | save_to_postgres.py |
| `anomalies` | Anomalies YoY > 20% | save_to_postgres.py |
| `rapports_nlp` | Rapports Mistral + structure JSON + scores | save_to_postgres.py |
| `nltk_analysis` | Analyse NLTK enrichie du rapport | save_to_postgres.py |

### Migration douce

`creer_tables()` utilise `ALTER TABLE ADD COLUMN IF NOT EXISTS` pour ajouter de nouvelles colonnes sans DROP, ce qui permet de tourner sur une base existante. Le renommage `created_at -> analyzed_at` (table nltk_analysis) est applique conditionnellement.

### Source unique de verite

Toutes les agregations filtrees partent de `ventes_detail`. Les tables `kpi_*` sont des caches Spark pre-calcules, utilisees uniquement quand aucun filtre n'est applique (gain de perfs).

### Seuil canonique des anomalies

Defini une seule fois dans `config.ANOMALY_THRESHOLD = 20.0` (en pourcentage). Applique :
- en Python par `src/nlp_transformers.detecter_anomalies`
- en SQL par `src/db.get_anomalies`
- dans les system prompts du chatbot
- dans le PDF (section anomalies)

Pas de divergence possible entre PDF, Grafana, API et chatbot.

---

## 8. Endpoints API et chatbot RAG

### Endpoints REST principaux

| Endpoint | Role |
|---|---|
| `GET /api/health` | Etat de la connexion DB + stats tables |
| `GET /api/filters` | Metadonnees des filtres disponibles |
| `GET /api/kpis/global` | KPIs globaux filtrables |
| `GET /api/kpis/annual` | Performance annuelle |
| `GET /api/kpis/regions` | Detail regions (avec ou sans summary) |
| `GET /api/kpis/categories` | Detail categories |
| `GET /api/kpis/quarterly` | Variations trimestrielles |
| `GET /api/kpis/monthly` | Ventes mensuelles |
| `GET /api/kpis/segments` | KPIs par segment |
| `GET /api/kpis/top-products` | Top sous-categories |
| `GET /api/kpis/top-clients` | Top clients |
| `GET /api/anomalies` | Anomalies detectees |
| `GET /api/nlp/report` | Rapport NLP (cache ou genere live) |
| `GET /api/nlp/quality` | Score qualite NLP detaille (Chantier 3) |
| `GET /api/nlp/analysis` | Analyse NLTK enrichie (Chantier 2) |
| `GET /api/rapport/structure` | Structure JSON hierarchisee (Chantier 1) |
| `POST /api/chat` | Chatbot RAG |
| `GET /api/pdf/generate` | Genere un PDF a la demande |
| `POST /api/reports/schedule` | Genere les 12 rapports en batch |
| `GET /api/dashboard` | Agregat de donnees pour le frontend |

Tous les endpoints de KPIs acceptent `year`, `region`, `category` en query parameters.

### Chatbot RAG

Architecture complete :

```
question utilisateur
   |
   v
detection injection (FORBIDDEN_PATTERNS)  --> refus generique si match
   |
   v
detection intention (top_client, anomalie, etc.)
   |
   v
construction contexte (KPI filtre + intent-specific)
   |
   v
embedding question (sentence-transformers 384d)
   |
   v
recherche semantique ChromaDB (top 5, seuil sim > 0.3)
   |
   v
appel Mistral chat (system prompt + contexte + question)
   |
   v
nettoyage markdown (strip_markdown)
   |
   v
reponse utilisateur (3-4 phrases, max 2 chiffres cles)
```

Le chatbot refuse les sujets hors retail. Le seuil 0.3 de similarite ChromaDB permet d'inclure les chunks moyennement pertinents pour enrichir le contexte sans bruit excessif.

---

## 9. Installation et execution

### Prerequis

- Python 3.9 ou plus
- Java 8 ou plus (pour Spark)
- PostgreSQL 14 ou plus
- Grafana (optionnel)

### Installation

```bash
git clone <repo_url>
cd NLP_REPORTING

python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows PowerShell
# OU
source venv/bin/activate         # Linux/Mac

pip install -r requirements.txt

cp .env.example .env             # puis editer avec vos credentials

createdb nlp_reporting
python load_detail_table.py
```

Fichier `.env` minimal :

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nlp_reporting
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe

MISTRAL_API_KEY=votre_cle_mistral
MISTRAL_MODEL=mistral-small-latest

API_HOST=127.0.0.1
API_PORT=8000
```

### Pipeline complet

```bash
python main.py                       # FR complet (defaut)
python main.py --langue en           # EN
python main.py --skip-db             # Sans PostgreSQL
python main.py --skip-pdf            # Sans PDF
python main.py --skip-db --skip-pdf  # Minimal (juste NLP en console)
```

Le pipeline execute les 7 etapes documentees plus haut et genere :
- 12 rapports NLP (1 global + 4 annees + 4 regions + 3 categories)
- 12 analyses NLTK (chacune sur le vrai texte Mistral)
- 1 PDF du rapport global
- Toutes les tables PostgreSQL peuplees

### API + dashboard web

```bash
python api.py
# http://127.0.0.1:8000        -> dashboard HTML
# http://127.0.0.1:8000/docs   -> Swagger interactif
```

### Indexation RAG

```bash
python load_rag_chunks.py
```

Construit les chunks depuis PostgreSQL, calcule les embeddings sentence-transformers (384d) et les insere dans ChromaDB (`.chroma_rag/`). A relancer apres chaque changement majeur de donnees.

### Grafana

Importer `grafana.json` dans Grafana, configurer la datasource PostgreSQL pointant sur `nlp_reporting`.

---

## 10. Tests et validation

### Tests pytest

```bash
pytest tests/ -v
```

Couvre :
- Configuration (import, chemins)
- NLTKProcessor (tokenisation, sentiment, keywords)
- Score business (calculer_score sur faits synthetiques)
- Anomalies (detecter_anomalies YoY)
- Resume executif

### Smoke test complet (DB et API reelles)

```bash
python tests/test_complet.py
```

Verifie :
- Imports critiques (mistralai, sentence-transformers, NLTK)
- Connexion PostgreSQL
- Transformer local
- Generation NLP
- Anomalies YoY
- Section NLTK PDF
- Endpoint scheduled

### Validation cache NLP

Si vous suspectez que des rapports en BDD contiennent des chiffres obsoletes (par exemple apres un changement de logique de calcul) :

```bash
python src/purger_cache_nlp.py          # dry-run, compte les suspects
python src/purger_cache_nlp.py --confirm # purge effective
python main.py                          # regenere tout proprement
```

---

## 11. Decisions architecturales et conventions

### Source unique de verite

Aucun composant ne recalcule un KPI : tous lisent depuis PostgreSQL. Si le PDF dit "CA = 2.2M", le dashboard, l'API et le chatbot disent exactement la meme chose.

### Resilience aux valeurs None

Croissance YoY = None quand pas d'annee comparable (premier exercice du perimetre). Toutes les fonctions de formatage gerent ce cas (helpers `_safe_float`, `_fmt_growth` dans `src/generate_pdf.py`).

### Filtres applique avant ou apres calcul

- `region` et `category` : appliques AVANT agregation. On calcule le YoY sur la trajectoire propre de la region ou de la categorie.
- `year` pour les anomalies : applique APRES le calcul du LAG, sinon le rapport annee N ne verrait plus l'annee N-1 et toutes les anomalies disparaitraient.

### Centralisation du markdown stripping

Une seule fonction `strip_markdown` dans `src/text_utils.py`, importee par `api.py`, `src/generate_pdf.py` et `src/nlp_transformers.py`. Avant la centralisation, 4 copies divergentes coexistaient.

### Coherence des embeddings RAG

`load_rag_chunks.py` et `api.py:_embed_question` utilisent le MEME modele (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions) pour eviter le mismatch dimensionnel dans ChromaDB.

### Migration douce du schema

`creer_tables()` applique des `ALTER TABLE ADD COLUMN IF NOT EXISTS` plutot que de DROP. Permet de tourner sur une base existante sans la casser.

### Conventions de code

- Pas d'emojis dans le code (commentaires, logs, strings utilisateur).
- Un docstring au debut de chaque fichier expliquant role, pourquoi, decisions.
- Pas de commentaires inline d'historique (les commits Git font foi).
- Type hints sur les signatures publiques.
- Logger Python standard, format unique `"%(asctime)s | %(levelname)-8s | %(message)s"`.

---

## 12. Limites connues et axes d'evolution

### Donnees

- Le dataset Superstore couvre 2015-2018, donc aucune prevision n'est possible au-dela. Le chatbot le mentionne explicitement quand on lui pose une question de forecast.
- 300 lignes du CSV ont des Sales nulles ou invalides, supprimees cote Spark mais pas cote `ventes_detail` (qui ingere les 9 800 brutes). Ecart minime sur les totaux globaux.

### NLP

- VADER reste actif en anglais mais peu adapte au B2B. La tonalite business par lexique metier le remplace pour le francais et est utilisee aussi en anglais dans le PDF.
- La detection d'intentions du chatbot est par mots-cles. Une evolution naturelle serait un classifieur fine-tune.

### PDF

- Les illustrations Unsplash necessitent un acces reseau au premier appel. Fallback vectoriel matplotlib si offline.
- Pas de table des matieres automatique sur les ~25 pages du PDF. A ajouter dans une version future.

### API

- Pas d'authentification : l'API est ouverte. A securiser en production via une cle d'API ou OAuth.
- Pas de rate limiting sur le chatbot. Un utilisateur abusif peut consommer des credits Mistral.

### Tests

- Tests d'integration limites (pas de fixtures DB dediees). En production il faudrait des fixtures pytest avec une base de donnees ephemere.

---

## Recapitulatif des 4 chantiers

| Chantier | Fichier principal | Apport |
|---|---|---|
| 1 - JSON structure | `src/nlp_transformers.py` | Anti-hallucination par validation contre les faits |
| 2 - Classification business | `src/nlp_nltk.py` | Score de couverture metier mesurable |
| 3 - Score qualite NLP | `nlp_quality_score.py` | Note /100 sur 6 criteres, comparaison entre runs |
| 4 - Recos hierarchisees | `src/nlp_transformers.py` + `src/generate_pdf.py` | Badges visuels priorite/niveau/confiance + justification chiffree |

---

## Licence

MIT. Projet academique ISM, libre d'usage pedagogique.
