# 📚 Rapport pédagogique — Mon projet NLP Reporting expliqué simplement

> Tout ce qu'il y a à comprendre dans le projet, expliqué comme à un enfant de 10 ans. Lis-le tranquillement, du début à la fin : à la fin, tu sauras expliquer le projet à n'importe qui.

---

## 🎯 1. C'est quoi ce projet, en une phrase ?

**Une machine automatique qui prend un fichier Excel rempli de ventes et qui produit, toute seule, un rapport business propre comme un humain l'écrirait.**

Imagine que tu travailles dans un magasin. Chaque jour, tu notes toutes tes ventes dans un grand cahier. À la fin du mois, ton patron te demande : *"Alors, ça a marché ? Qu'est-ce qui s'est bien vendu ? Où est le problème ?"*. Tu vas devoir :

1. **Compter** toutes les ventes
2. **Faire le tri** par catégorie, par région, par mois
3. **Repérer ce qui est bizarre** (une chute, un pic)
4. **Écrire un résumé** en bon français
5. **Faire un joli document** pour ton patron

C'est long. Très long. Mon projet **fait tout ça tout seul, en 5 minutes**.

---

## 🏗️ 2. La grande chaîne de production (le pipeline)

Imagine une usine de saucisses : il y a plusieurs machines en ligne. La première fait une chose, la deuxième fait autre chose, etc. À la fin, la saucisse est prête. Mon projet, c'est pareil, sauf qu'au lieu de saucisses ce sont des rapports.

Voici les 7 étapes (les 7 machines de l'usine) :

```
[CSV de ventes brutes]
        │
        ▼
┌────────────────────┐
│  1. SPARK          │  Lit le fichier, le nettoie, calcule 12 indicateurs
└────────────────────┘
        │
        ▼
┌────────────────────┐
│  2. NLTK (extract) │  Transforme les chiffres en "faits" structurés
└────────────────────┘
        │
        ▼
┌────────────────────┐
│  3. MISTRAL JSON   │  Un robot écrivain (Transformer) fabrique un plan JSON
└────────────────────┘
        │
        ▼
┌────────────────────┐
│  4. MISTRAL TEXTE  │  Le même robot transforme le plan en texte fluide
└────────────────────┘
        │
        ▼
┌────────────────────┐
│  5. NLTK (analyse) │  Vérifie la qualité du texte généré
└────────────────────┘
        │
        ▼
┌────────────────────┐
│  6. POSTGRESQL     │  Range tout dans une grosse boîte de rangement (la base)
└────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  7. SORTIES                            │
│   • PDF                                │
│   • Dashboard Grafana                  │
│   • API web                            │
│   • Chatbot qui répond aux questions   │
└────────────────────────────────────────┘
```

On va maintenant zoomer sur chaque étape pour vraiment comprendre.

---

## 🔥 3. Spark — Le calculateur géant

### 3.1 — C'est quoi Spark ?

**Spark, c'est un programme qui sait compter très, très vite, même sur des fichiers énormes.**

Pour comprendre, imagine :
- **Excel** : sait gérer 1 million de lignes max, et il rame déjà.
- **Pandas** (autre outil Python) : sait gérer jusqu'à quelques dizaines de millions de lignes.
- **Spark** : sait gérer des **milliards** de lignes, en partageant le travail entre plusieurs ordinateurs.

C'est comme la différence entre :
- Toi tout seul qui comptes des billes (Excel)
- Toi et un copain (Pandas)
- Toi et 100 copains qui se partagent le tas (Spark)

> **Pourquoi je l'utilise alors que mon fichier ne fait que 9 800 lignes ?**
> Parce que le code que j'écris fonctionnera **exactement pareil** demain si on me donne 9 millions de lignes. Pas besoin de réécrire. C'est ce qu'on appelle **"scalable"** (qui peut grandir).

### 3.2 — Ce que Spark fait dans MON projet

**Étape A — Charger le fichier CSV** (fichier `src/load_data.py`) :

Spark ouvre le fichier `data/train.csv` (9 800 lignes de ventes du magasin Superstore, années 2015 à 2018). Il fait ensuite **le ménage** :

| Ce que Spark fait | Pourquoi |
|---|---|
| Renomme `Order ID` → `Order_ID` | Les espaces dans les noms causent des bugs |
| Convertit `Order_Date` (texte "03/01/2017") en vraie date | Pour pouvoir filtrer par année/mois |
| Supprime les lignes où `Sales <= 0` ou nulle | Vente à 0 €, ça ne veut rien dire |
| Supprime les lignes avec date invalide | Pour éviter les erreurs plus tard |
| Ajoute 6 nouvelles colonnes : `Annee`, `Trimestre`, `Mois`, `Semaine`, `Jour_Semaine`, `Delai_Livraison` | Pour pouvoir faire des analyses temporelles |

À la fin de l'étape A, Spark a un grand tableau bien propre, prêt à être analysé.

**Étape B — Calculer 12 KPIs** (fichier `src/analytics.py`) :

Un **KPI** = un **indicateur clé de performance** (Key Performance Indicator). C'est un chiffre important qui dit comment va le business. Mon projet en calcule 12 :

| # | KPI | Ce que ça veut dire | Technique Spark utilisée |
|---|---|---|---|
| 1 | **Ventes par région et trimestre** | Combien on a vendu dans l'Ouest au T1 2017 ? | `groupBy + agg` (groupage + agrégation) |
| 2 | **Ventes par catégorie et région** | Combien de "Furniture" vendu dans le Sud ? | `groupBy + pivot` |
| 3 | **Top 10 sous-catégories** | Les 10 produits qui rapportent le plus | `groupBy + orderBy + limit` |
| 4 | **Variations trimestrielles QoQ et YoY** | Le T2 a fait +15% par rapport au T1, ou +20% par rapport au T2 de l'année dernière | **Window + lag** |
| 5 | **Performance annuelle avec YoY** | 2018 a fait +40% par rapport à 2017 | Window + lag |
| 6 | **Meilleure région par année** | En 2017, c'est l'Ouest qui a le mieux marché | Window + rank |
| 7 | **Ventes mensuelles + moyenne mobile 3M** | Janvier : 50k$, Février : 60k$, et la moyenne lissée sur 3 mois pour gommer les pics | Window + avg over rows |
| 8 | **Ventes par segment client** | Combien achètent les "Consumer" vs "Corporate" vs "Home Office" | groupBy |
| 9 | **Performance par mode de livraison** | Same Day, First Class, Standard Class : qui marche le mieux | groupBy + agg |
| 10 | **Top 10 clients** | Les 10 clients qui dépensent le plus chez nous | groupBy + agg + rank |
| 11 | **Analyse par État américain** | Quels États rapportent le plus (Californie, Texas, etc.) | groupBy |
| 12 | **Saisonnalité par jour de semaine** | On vend plus le lundi ou le vendredi ? | groupBy + dayofweek |

### 3.3 — Les concepts Spark utilisés (le vocabulaire à connaître)

**👉 `groupBy`** : "regrouper par".
> Exemple : `df.groupBy("Region").sum("Sales")` → "Donne-moi la somme des ventes pour chaque région".

**👉 `agg` (aggregate)** : "agrégrer" = calculer un résumé (somme, moyenne, max, count...).

**👉 `Window function`** : une fonction qui regarde "autour" de chaque ligne pour calculer quelque chose.
> Exemple : pour calculer "combien j'ai vendu en plus que le mois dernier", je dois pouvoir regarder le mois précédent. C'est le rôle de `lag()` (= "retard", aller chercher la valeur précédente).

**👉 `countDistinct`** : compter les **uniques**.
> Très important : dans le CSV, une commande peut être sur plusieurs lignes (un client achète 3 articles = 3 lignes mais 1 seule commande). `countDistinct(Order_ID)` compte les vraies commandes, pas les lignes.

**👉 Convention métier importante du projet** :
- `Nb_Commandes = countDistinct(Order_ID)` → les vraies commandes
- `Nb_Articles = count(*)` → les lignes (= articles d'une commande)
- `Panier_Moyen = SUM(Sales) / Nb_Commandes` → panier moyen **par commande** (pas par ligne)

Cette convention est appliquée **partout** dans le projet (Spark, BDD, API, PDF) → pas de divergence.

### 3.4 — Qu'est-ce que Spark sait écrire après ?

Spark sauvegarde au format **Parquet** (un format compressé super rapide à lire), partitionné par année et région. Mais dans mon projet, on garde surtout les résultats en mémoire pour les passer à l'étape NLP.

---

## 🧠 4. NLTK — Le linguiste du projet

### 4.1 — C'est quoi NLTK ?

**NLTK = "Natural Language Toolkit" = la boîte à outils pour la langue humaine.**

C'est un programme Python qui sait **lire, découper et analyser du texte en langue humaine** (anglais, français, etc.). C'est comme un dictionnaire intelligent qui sait :

- Couper un texte en phrases
- Couper une phrase en mots
- Reconnaître les noms propres ("Paris", "Apple", "Marie")
- Savoir si un mot est un nom, un verbe, un adjectif…
- Compter les mots les plus fréquents
- Reconnaître si une phrase est positive ou négative

C'est la **boîte à outils classique** du NLP (Natural Language Processing = traitement du langage). Elle existe depuis 2001, et c'est ce qu'on apprend à l'école quand on étudie le traitement de texte.

### 4.2 — Les 6 outils de NLTK que j'utilise

1. **Tokenization (`sent_tokenize`, `word_tokenize`)** : couper le texte en phrases et en mots.
   > "Le chat dort. Il mange." → ["Le chat dort.", "Il mange."] → ["Le", "chat", "dort"], ["Il", "mange"]

2. **Stopwords** : enlever les mots inutiles ("le", "la", "et", "de"…).
   > Parce que pour analyser, "Le chat dort" devient "chat dort" → plus efficace.

3. **Lemmatization (`WordNetLemmatizer`)** : ramener les mots à leur forme de base.
   > "mangeaient" → "manger" ; "chats" → "chat". C'est comme dans le dictionnaire.

4. **POS tagging (`pos_tag`)** : étiqueter chaque mot avec sa nature grammaticale.
   > "Le chat dort" → ("Le", DT/déterminant), ("chat", NN/nom), ("dort", VBZ/verbe)

5. **NER (Named Entity Recognition) (`ne_chunk`)** : reconnaître les noms propres.
   > "Marie vit à Paris" → PERSON: Marie, GPE: Paris (GPE = lieu géopolitique)

6. **VADER sentiment** : dire si une phrase est positive, négative ou neutre (anglais seulement).
   > "I love this product" → positif. "This is terrible" → négatif.

### 4.3 — Comment j'utilise NLTK dans MON projet

NLTK intervient à **2 endroits différents** :

#### A — AVANT Mistral : extraire les "faits"

Fichier : `src/nlp_nltk.py`, fonction `extraire_faits(kpis)`.

À ce stade, Spark a produit ses 12 KPIs (gros tableaux de chiffres). Mais on ne peut pas balancer ça brut au robot écrivain Mistral — ce serait illisible.

Donc NLTK **transforme les chiffres en faits structurés** (un grand dictionnaire Python) :

```python
faits = {
    "annuel": [{"annee": 2017, "ca": 600000, "yoy": 25.3}, ...],
    "meilleure_annee": {"annee": 2018, "ca": 713927},
    "pire_annee": {"annee": 2015, "ca": 478000},
    "meilleur_trim": {"trim": "T4 2017", "var_yoy": 69.45},
    "pire_trim": {"trim": "T1 2016", "var_yoy": -64.9},
    "top_produits": [{"nom": "Phones", "ventes": 327528}, ...],
    "anomalies": [...],
    "volatilite": {...},
    ...
}
```

Ce dictionnaire est **propre, structuré, vérifiable**. C'est lui qu'on donne ensuite à Mistral.

#### B — APRÈS Mistral : vérifier la qualité du texte généré

Fichier : `src/nlp_nltk.py`, fonction `analyser_rapport_genere(texte, langue)`.

Une fois que Mistral a écrit le rapport en prose, NLTK le **relit pour vérifier la qualité** :

- **Couverture business (Chantier 2)** : chaque phrase est classée par "intention" :
  - `tendance` (décrit ce qui se passe)
  - `anomalie` (signale un écart)
  - `opportunité` (identifie un levier)
  - `risque` (alerte)
  - `recommandation` (propose une action)
  - `contexte` (le reste)

  → On vérifie qu'un bon rapport contient au moins 1 tendance, 1 alerte, 2 recos, 1 opportunité.

- **Tonalité business** : on compte les mots positifs vs négatifs du **lexique métier** (croissance, expansion, risque, fragilité...). Donne un score de -100 (très négatif) à +100 (très positif).

  > 💡 Pourquoi pas VADER ? Parce que VADER est anglais-only et calibré pour Twitter. Il dit que "forte chute" est POSITIF à cause du mot "forte". Pour le B2B et le français, on a un lexique métier dédié.

- **Thèmes dominants** : on compte les occurrences des 12 thèmes (croissance, décroissance, saisonnalité, concentration produit, volatilité, fidélisation, panier moyen, diversification, risque, opportunité, recommandation, concentration régionale). Les 3 plus présents sont les thèmes dominants du rapport.

---

## 🤖 5. Transformer & Mistral — Le robot écrivain

### 5.1 — C'est quoi un Transformer ?

**Un Transformer, c'est une famille de programmes super-intelligents capables de comprendre ET d'écrire du texte comme un humain.**

Imagine un perroquet qui aurait lu 100 millions de livres. Il connaît :
- Les mots
- La grammaire
- Les expressions
- Les contextes
- Et il peut combiner tout ça pour écrire un texte original sur n'importe quel sujet.

Le Transformer a été **inventé en 2017** par Google dans un article célèbre intitulé "Attention Is All You Need". L'idée révolutionnaire : au lieu de lire le texte mot par mot dans l'ordre, le Transformer regarde **tous les mots en même temps** et identifie **lesquels sont les plus importants** pour comprendre le sens (c'est l'**attention mechanism**).

> **Tu connais peut-être ChatGPT** ? Eh bien ChatGPT est un Transformer. **GPT** signifie d'ailleurs **"Generative Pre-trained Transformer"**.

### 5.2 — Mistral, c'est quoi ?

**Mistral, c'est un Transformer fait par une entreprise française (Mistral AI, fondée par d'anciens ingénieurs de Meta et Google).**

C'est l'équivalent français de ChatGPT. Plus précisément :
- Mistral AI propose **un service en ligne** : on envoie une question/un texte via internet → on récupère la réponse.
- Le modèle que j'utilise dans le projet s'appelle `mistral-small-latest`.
- On y accède via une **clé API** (un mot de passe secret stocké dans le fichier `.env`).

### 5.3 — Comment j'utilise Mistral dans MON projet (en 2 étapes)

C'est là où mon projet devient intéressant — j'ai conçu une **architecture en 2 étapes pour éviter les "hallucinations"** (= quand un LLM invente des chiffres faux).

Fichier : `src/nlp_transformers.py`.

#### Étape 1 — Demander un JSON structuré

**Au lieu de** demander directement à Mistral : *"Écris-moi un rapport sur ces ventes"* (= risqué, il pourrait inventer des chiffres),

**je demande** : *"À partir de ces faits exacts (que NLTK a extraits), remplis ce template JSON avec : un résumé, les risques, les opportunités, et des recommandations. NE RAJOUTE AUCUN CHIFFRE QUE JE NE T'AI PAS DONNÉ."*

Mistral renvoie alors un JSON comme celui-ci :

```json
{
  "resume": "L'année 2018 marque un record avec 713k$ ...",
  "tendance_globale": "croissance soutenue",
  "risques": [
    {
      "titre": "Volatilité trimestrielle élevée",
      "description": "Écart-type de 30% sur les variations QoQ",
      "criticite": "moyenne"
    }
  ],
  "opportunites": [...],
  "recommandations": [
    {
      "action": "Diversifier la concentration sur Phones (45% du CA)",
      "priorite": "haute",
      "niveau": "strategique",
      "confiance": "elevee",
      "justification": "Phones = 327k$ (45%) → si chute, impact majeur"
    }
  ]
}
```

Ce JSON est ensuite **validé** : on vérifie qu'aucun chiffre inventé n'a été glissé. Si Mistral écrit "Le CA est de 800k$", on regarde si 800k$ correspond à un vrai fait dans notre dictionnaire. Si non → on remplace par le vrai chiffre.

#### Étape 2 — Transformer le JSON en prose narrative

On rappelle Mistral et on lui dit : *"Voici ton JSON structuré. Maintenant transforme-le en 4 paragraphes de texte fluide (~400-600 mots). Pas de markdown, pas de bullet points. Du beau français business."*

Mistral renvoie alors un texte comme :

> *"L'année 2018 marque un point culminant pour Superstore avec un chiffre d'affaires de 713 927 USD, en hausse de 49,5% par rapport à 2017. Cette dynamique est portée par la catégorie Phones, qui réalise 327 528 USD à elle seule. Toutefois, cette performance masque une volatilité préoccupante : le T1 2016 a connu une chute brutale de 64,9%, illustrant la fragilité saisonnière du business..."*

C'est ce texte qui se retrouve **dans le PDF, sur le dashboard, et dans le chatbot**.

### 5.4 — Pourquoi cette architecture en 2 étapes ?

| Problème | Solution apportée |
|---|---|
| Le LLM invente des chiffres ("hallucination") | On lui donne un JSON validé, il ne fait que l'habiller en prose |
| Les chiffres ne correspondent pas à ceux du PDF | Une seule source : le JSON validé |
| Si Mistral plante, on n'a rien | Fallback : on génère une prose simple à partir du JSON |
| Difficile de garantir la structure | Le JSON force la structure (résumé + risques + opportunités + recos) |

### 5.5 — Les 4 "Chantiers" NLP du projet

Mon projet ne se contente pas de "générer un texte". Il ajoute 4 surcouches intelligentes :

#### Chantier 1 — JSON structuré (anti-hallucination)
✅ Décrit ci-dessus. **Garantit que les chiffres dans le texte = chiffres dans Spark.**

#### Chantier 2 — Classification d'intentions business
✅ Décrit dans la section NLTK. **Mesure la couverture business du rapport (tendance / risque / reco / opportunité).**

#### Chantier 3 — Score qualité NLP /100
Fichier : `nlp_quality_score.py`.

**6 critères pondérés** pour noter un rapport généré :

| Critère | Poids | Ce qu'on vérifie |
|---|---|---|
| Couverture des faits | 25 pts | Le texte cite-t-il la meilleure année, le pire trimestre, le top produit ? |
| Ancrage numérique | 25 pts | Combien de chiffres concrets ($, %) dans le texte ? |
| Présence de recommandations | 15 pts | Au moins 2 actions concrètes ? |
| Clarté / lisibilité | 15 pts | Longueur moyenne des phrases, ratio de mots uniques |
| Absence de répétition | 10 pts | Diversité des bigrammes |
| Ton business | 10 pts | Vocabulaire métier présent, pas de mots vagues |

Le score est stocké en base et affiché partout. Permet de **comparer deux runs** : "le nouveau prompt a fait monter le score de 75 à 82". Très pratique pour l'amélioration continue.

#### Chantier 4 — Recommandations hiérarchisées avec badges

Chaque recommandation a 3 attributs :
- **Priorité** : `haute` / `moyenne` / `basse` (= ce qu'on fait en premier)
- **Niveau** : `strategique` / `tactique` / `operationnel` (= qui s'en occupe : CODIR / manager / équipe)
- **Confiance** : `elevee` / `moyenne` / `faible` (= dérivée automatiquement de la volatilité des données)

Plus une **justification chiffrée obligatoire**.

Les recos sont **triées automatiquement par priorité** et rendues dans le PDF avec des **badges colorés** (rouge / orange / vert).

---

## 🗄️ 6. PostgreSQL — Le grand classeur

### 6.1 — C'est quoi PostgreSQL ?

**PostgreSQL, c'est un système de rangement super-organisé pour stocker plein de données.**

Imagine un grand classeur avec plein de dossiers. Chaque dossier est un **tableau** (une *table*) avec des **lignes** et des **colonnes**. Tu peux à tout moment dire : "Donne-moi toutes les ventes de 2017 supérieures à 1000$" → PostgreSQL te répond en quelques millisecondes.

C'est **gratuit**, c'est utilisé par Apple, Spotify, Instagram, et c'est la référence académique.

### 6.2 — Mes 12 tables PostgreSQL

| Table | Ce qu'elle contient |
|---|---|
| `ventes_detail` | Toutes les lignes du CSV brut (9 800 lignes), grain transactionnel |
| `kpi_global` | Le récap global (CA total, croissance globale, top région) |
| `kpi_annuel` | 1 ligne par année avec le CA et la croissance YoY |
| `kpi_region_trim` | Ventes par région × trimestre |
| `kpi_variation` | Variations trimestrielles QoQ |
| `kpi_top_produits` | Le top 10 sous-catégories |
| `kpi_mensuel` | Ventes par mois + moyenne mobile |
| `kpi_segment` | Ventes par segment client |
| `kpi_categorie` | Ventes catégorie × région × année |
| `anomalies` | Les anomalies YoY > 20% détectées automatiquement |
| `rapports_nlp` | Les rapports Mistral + leur structure JSON + leur score |
| `nltk_analysis` | L'analyse NLTK enrichie du rapport (tonalité, thèmes, etc.) |

### 6.3 — Le principe fondamental : "Source unique de vérité"

C'est une règle d'or du projet : **personne ne recalcule un KPI**. Tout le monde lit dans la base PostgreSQL.

> Si le PDF dit "CA = 2,2M", le dashboard Grafana dit "CA = 2,2M", l'API dit "CA = 2,2M" et le chatbot dit "CA = 2,2M". **Aucune divergence possible.**

C'est pour ça qu'on a écrit `src/save_to_postgres.py` qui range tout proprement après le calcul Spark.

---

## 📤 7. Les sorties du projet

Une fois que tout est calculé et rangé, on peut produire 4 sorties différentes.

### 7.1 — Le PDF (le livrable phare)

Fichier : `src/generate_pdf.py` + `src/pdf_design_professional.py`.

C'est un PDF de ~25 pages, généré automatiquement avec **ReportLab** (lib Python pour faire des PDF) et **Matplotlib** (lib pour faire des graphiques).

Il contient :
- Page de garde
- Résumé exécutif
- KPIs principaux en cartes colorées
- Graphiques (barres, lignes, camemberts)
- Section NLTK : tonalité business, thèmes dominants
- Section Mistral : le rapport narratif
- Section Recommandations : avec **badges colorés priorité/niveau/confiance** (Chantier 4)
- Section Anomalies : YoY > 20%
- Section Score NLP : avec le détail des 6 critères

Le PDF inclut même des **photos Unsplash** (illustrations professionnelles téléchargées en ligne, mises en cache local).

### 7.2 — Le dashboard Grafana

Fichier : `grafana.json`.

**Grafana**, c'est un outil web super puissant pour **visualiser des données** sous forme de graphiques interactifs. On le connecte directement à PostgreSQL et on bâtit des dashboards avec des filtres.

Le dashboard du projet a (dans sa version originale) :
- **6 KPIs en haut** (CA, Commandes, Croissance, Panier Moyen, Clients Uniques, Score NLP)
- **5 sections déroulantes** : 🤖 NLP & Insights / 🌍 Répartitions / 🏆 Classements / 📊 Variations & Performance / 📈 Évolution Temporelle
- **4 filtres** en haut : Année, Région, Segment, Catégorie
- **17 panels** au total

### 7.3 — L'API REST (FastAPI)

Fichier : `api.py`.

**Une API**, c'est une porte d'entrée par laquelle d'autres programmes peuvent demander des informations à mon projet. Au lieu d'ouvrir Grafana ou le PDF, un autre programme peut juste faire `GET http://localhost:8000/api/kpis/global` et récupérer les KPIs en JSON.

L'API a **plus de 20 endpoints** : `/api/health`, `/api/kpis/global`, `/api/kpis/annual`, `/api/anomalies`, `/api/nlp/report`, `/api/chat` (chatbot), etc.

Documentation auto-générée disponible sur `/docs` (Swagger).

### 7.4 — Le chatbot RAG

C'est la cerise sur le gâteau.

**RAG = "Retrieval-Augmented Generation"** = "Génération augmentée par recherche".

Le principe :
1. L'utilisateur pose une question : *"Quelle est ma meilleure région en 2018 ?"*
2. La question est **transformée en vecteur** (un tableau de 384 nombres) par **sentence-transformers** (un mini-Transformer spécialisé pour ça).
3. Ce vecteur est comparé à **tous les "chunks" du projet** (morceaux de texte des rapports, KPIs, anomalies) stockés dans **ChromaDB** (une base vectorielle).
4. Les 5 chunks les plus pertinents sont récupérés.
5. On envoie à Mistral : *"Voici la question, voici les 5 chunks pertinents, voici les KPIs filtrés. Réponds en 3-4 phrases business."*
6. Mistral répond, on nettoie le markdown, et on envoie à l'utilisateur.

Le chatbot **refuse les sujets hors retail** (sécurité), détecte les **injections de prompt** et ne répond pas aux questions de forecast (pas de données après 2018).

---

## 🛠️ 8. Stack technique complète (récap)

| Couche | Technologie | Définition simple | Utilisation dans le projet |
|---|---|---|---|
| Données | **Apache Spark** (PySpark) | Calculateur distribué pour gros volumes | Chargement, nettoyage, 12 KPIs |
| Stockage | **PostgreSQL** | Base de données relationnelle | 12 tables, source unique de vérité |
| Vecteurs | **ChromaDB** | Base vectorielle pour la recherche sémantique | Stockage des chunks RAG |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Mini-Transformer qui transforme texte → vecteur 384d | Encodage des chunks ET des questions |
| NLP classique | **NLTK** | Boîte à outils linguistique | Extraction faits + analyse rapport |
| NLP génération | **Mistral API** (`mistral-small-latest`) | LLM Transformer en cloud | Génération JSON + prose narrative |
| Score qualité | Python custom | 6 critères pondérés | Note /100 par rapport |
| API web | **FastAPI** | Framework Python pour APIs REST | 20+ endpoints + chatbot |
| Dashboard | **Grafana** | Visualisation interactive de données | Connecté à PostgreSQL |
| PDF | **ReportLab + Matplotlib** | Génération de PDF pro avec graphiques | 25 pages auto |
| Frontend | HTML/CSS/JS + Chart.js | Web simple sans framework | Page dashboard + chatbot |
| Tests | **pytest** | Framework de tests Python standard | Tests unitaires + smoke test |

---

## 📊 9. Les 4 Chantiers NLP — récapitulatif

| Chantier | Fichier | Apport métier | Pourquoi c'est génial |
|---|---|---|---|
| **1 — JSON structuré** | `nlp_transformers.py` | Anti-hallucination par validation contre les faits | Les chiffres dans le texte sont GARANTIS exacts |
| **2 — Classification business** | `nlp_nltk.py` | Score de couverture business mesurable | On sait si le rapport parle bien de tendances, risques, recos |
| **3 — Score qualité /100** | `nlp_quality_score.py` | Note objective sur 6 critères | On peut comparer 2 runs, détecter une régression |
| **4 — Recos hiérarchisées** | `nlp_transformers.py` + `generate_pdf.py` | Badges visuels priorité/niveau/confiance + justification chiffrée | Une reco devient actionnable, pas vague |

---

## 🎓 10. Si on me pose la question à la soutenance…

### *"Pourquoi tu as choisi Spark alors que 9 800 lignes ?"*
Parce que le code est conçu pour **scaler**. Demain à 9 millions de lignes, **rien ne change**. C'est une preuve de maîtrise d'une stack production-grade, pas un choix d'amateur.

### *"Pourquoi Mistral et pas ChatGPT ?"*
- **Mistral est français** → souveraineté numérique
- **Architecture Transformer** identique à GPT
- **API simple** et coût raisonnable
- **Réponses en français de qualité** (pas de traduction approximative)

### *"Qu'est-ce qui empêche les hallucinations ?"*
Le **Chantier 1 — pipeline en 2 étapes** :
1. Mistral produit d'abord un **JSON validé contre les faits NLTK**
2. Puis transforme ce JSON en prose
→ Il ne **re-calcule jamais** un chiffre, il **habille** des faits.

### *"Quelle est la valeur ajoutée du Score NLP /100 ?"*
- **Comparer deux runs** : *"Mon nouveau prompt a fait passer le score de 75 à 82"*
- **Détecter une régression** : si demain le score chute, on sait que quelque chose s'est dégradé
- **Mesurable et reproductible** : 6 critères pondérés, pas de subjectivité

### *"Pourquoi NLTK et pas spaCy ou un modèle Hugging Face ?"*
- **NLTK couvre toute la chaîne classique** : tokenisation, NER, POS, sentiment
- C'est la **référence pédagogique** du NLP
- spaCy serait plus rapide mais on perd la dimension "boîte à outils classique" du programme

### *"Pourquoi ChromaDB et pas pgvector ?"*
- **Pas besoin d'extension Postgres** (plus simple à déployer)
- **Process Python natif**, persiste sur disque
- **Adapté au volume** : quelques milliers de chunks, pas besoin de scalabilité massive

### *"Comment tu garantis que PDF, Grafana, API disent la même chose ?"*
**Source unique de vérité = PostgreSQL.** Personne ne recalcule. Le seuil d'anomalies (`20%`) est défini **une seule fois** dans `config.ANOMALY_THRESHOLD` et utilisé partout (Python, SQL, prompts du chatbot, PDF).

---

## 🚀 11. Comment lancer le projet (pense-bête)

```bash
# 1. Activer l'environnement
.\venv\Scripts\Activate.ps1

# 2. Lancer le pipeline complet (FR par défaut)
python main.py

# 3. Variantes
python main.py --langue en           # version anglaise
python main.py --skip-db             # sans PostgreSQL
python main.py --skip-pdf            # sans PDF

# 4. API + dashboard web
python api.py
# Ouvre http://127.0.0.1:8000 → dashboard HTML
# Ouvre http://127.0.0.1:8000/docs → Swagger interactif

# 5. Indexation RAG (à refaire si données changent)
python load_rag_chunks.py

# 6. Tests
pytest tests/ -v
python tests/test_complet.py  # smoke test complet
```

---

## 💡 12. La "morale" du projet

Tu n'as pas juste **collé des outils ensemble**. Tu as résolu **3 vrais problèmes** :

1. **Le problème de la lenteur** : Spark calcule en 2 secondes ce qu'un humain mettrait 3 jours à faire.
2. **Le problème de la qualité** : grâce à NLTK + Mistral + le Score /100, le rapport est **bon, mesurable, comparable**.
3. **Le problème de la cohérence** : grâce à PostgreSQL = source unique, **tous les livrables disent la même chose**.

C'est de l'**ingénierie data + NLP + produit**. Tu peux le défendre en stage, en entretien, ou même le déployer en vrai dans une PME qui a besoin de reporting automatique.

🎉 **Bravo.**
