# Documentation API du système de matching CV-Offres d'emploi

## Introduction

Cette API permet de gérer un système de matching entre des candidatures (CV) et des offres d'emploi. Elle utilise des techniques d'extraction d'information, d'analyse sémantique et de correspondance de compétences pour identifier les meilleurs candidats pour une offre donnée.

## Endpoints

### Gestion des offres d'emploi

#### `POST /api/job-offers`

Créer une nouvelle offre d'emploi.

**Options de requête :**
- Format JSON : Envoyer les détails de l'offre directement au format JSON
- Format multipart/form-data : Envoyer un fichier PDF ou image (`job_document` ou `offer`) contenant la description de l'offre

**Paramètres (JSON) :**
- `title` (obligatoire) : Titre du poste
- `company` (obligatoire) : Nom de l'entreprise
- `description` (optionnel) : Description du poste
- `location` (optionnel) : Localisation du poste
- `level` (optionnel) : Niveau d'expérience requis
- `keywords` (optionnel) : Liste des compétences requises

**Paramètres (multipart/form-data) :**
- `job_document` ou `offer` (obligatoire) : Fichier PDF ou image du descriptif de poste
- `title` (optionnel) : Titre du poste
- `company` (optionnel) : Nom de l'entreprise
- `location` (optionnel) : Localisation du poste
- `level` (optionnel) : Niveau d'expérience requis
- `job_text` (optionnel) : Texte supplémentaire pour la description

**Réponse :**
```json
{
  "job_id": "6800fa7847bc1978f0f9f974",
  "job_offer": {
    "_id": "6800fa7847bc1978f0f9f974",
    "title": "Développeur FullStack",
    "company": "TechSolutions",
    "description": "...",
    "keywords": ["python", "flask", "react", "mongodb"],
    "created_at": "2023-10-30T13:45:33.126Z"
  }
}
```

#### `GET /api/job-offers`

Récupérer la liste de toutes les offres d'emploi.

**Réponse :**
```json
[
  {
    "_id": "6800fa7847bc1978f0f9f974",
    "title": "Développeur FullStack",
    "company": "TechSolutions",
    "description": "...",
    "keywords": ["python", "flask", "react", "mongodb"],
    "created_at": "2023-10-30T13:45:33.126Z"
  },
  {
    "_id": "6800fc2c47bc1978f0f9f975",
    "title": "Data Scientist",
    "company": "AI Solutions",
    "description": "...",
    "keywords": ["python", "machine learning", "tensorflow", "pandas"],
    "created_at": "2023-10-30T13:51:24.763Z"
  }
]
```

#### `GET /api/job-offers/{offer_id}`

Récupérer les détails d'une offre d'emploi spécifique.

**Paramètres :**
- `offer_id` : Identifiant de l'offre d'emploi

**Réponse :**
```json
{
  "_id": "6800fa7847bc1978f0f9f974",
  "title": "Développeur FullStack",
  "company": "TechSolutions",
  "description": "...",
  "keywords": ["python", "flask", "react", "mongodb"],
  "created_at": "2023-10-30T13:45:33.126Z"
}
```

### Gestion des candidatures

#### `POST /api/apply`

Soumettre une candidature à une offre d'emploi spécifique.

**Paramètres (multipart/form-data) :**
- `cv` (obligatoire) : Fichier CV au format PDF, DOCX, TXT ou image
- `offer_id` (obligatoire) : Identifiant de l'offre d'emploi

**Réponse :**
```json
{
  "application_id": "68017f6a8f8d0455b75f43dd",
  "candidate": {
    "personal_info": {
      "name": "Jean Dupont",
      "email": "jean.dupont@example.com",
      "telephone": "+33612345678"
    },
    "optimized_resume": "Profil professionnel:\n...",
    "skills": ["python", "flask", "mongodb", "docker"]
  },
  "job_offer": {
    "id": "6800fa7847bc1978f0f9f974",
    "title": "Développeur FullStack",
    "company": "TechSolutions",
    "keywords": ["python", "flask", "react", "mongodb"]
  },
  "matching": {
    "score": 73.5,
    "skill_match_score": 85.2,
    "semantic_similarity": 62.8,
    "matched_skills": ["python", "flask", "mongodb"]
  }
}
```

#### `POST /api/apply-open`

Soumettre une candidature spontanée (sans référence à une offre spécifique).

**Paramètres (multipart/form-data) :**
- `cv` (obligatoire) : Fichier CV au format PDF, DOCX, TXT ou image

**Réponse :**
```json
{
  "application_id": "68017f6a8f8d0455b75f43dd",
  "candidate": {
    "personal_info": {
      "name": "Jean Dupont",
      "email": "jean.dupont@example.com",
      "telephone": "+33612345678"
    },
    "optimized_resume": "Profil professionnel:\n...",
    "skills": ["python", "flask", "mongodb", "docker"]
  },
  "matching_offers": [
    {
      "offer_id": "6800fa7847bc1978f0f9f974",
      "title": "Développeur FullStack",
      "company": "TechSolutions",
      "score": 73.5,
      "matched_skills": ["python", "flask", "mongodb"]
    },
    {
      "offer_id": "6800fc2c47bc1978f0f9f975",
      "title": "Développeur Backend",
      "company": "WebTech",
      "score": 65.2,
      "matched_skills": ["python", "flask"]
    }
  ]
}
```

#### `GET /api/postulations`

Récupérer la liste de toutes les candidatures.

**Réponse :**
```json
[
  {
    "_id": "68017f6a8f8d0455b75f43dd",
    "offer_id": "6800fa7847bc1978f0f9f974",
    "personal_info": {
      "name": "Jean Dupont",
      "email": "jean.dupont@example.com"
    },
    "date_applied": "2023-10-30T14:22:18.763Z",
    "match_data": {
      "final_score": 73.5
    }
  },
  {
    "_id": "68017f6d8f8d0455b75f43de",
    "offer_id": "6800fc2c47bc1978f0f9f975",
    "personal_info": {
      "name": "Marie Martin",
      "email": "marie.martin@example.com"
    },
    "date_applied": "2023-10-30T14:22:21.107Z",
    "match_data": {
      "final_score": 68.2
    }
  }
]
```

#### `GET /api/postulations/{postulation_id}`

Récupérer les détails d'une candidature spécifique.

**Paramètres :**
- `postulation_id` : Identifiant de la candidature

**Réponse :**
```json
{
  "_id": "68017f6a8f8d0455b75f43dd",
  "offer_id": "6800fa7847bc1978f0f9f974",
  "cv_text": "...",
  "entities": {
    "name": "Jean Dupont",
    "email": "jean.dupont@example.com",
    "telephone": "+33612345678",
    "skills": ["python", "flask", "mongodb", "docker"]
  },
  "personal_info": {
    "name": "Jean Dupont",
    "email": "jean.dupont@example.com",
    "telephone": "+33612345678"
  },
  "optimized_resume": "Profil professionnel:\n...",
  "match_data": {
    "skill_match_score": 85.2,
    "semantic_similarity": 62.8,
    "final_score": 73.5,
    "matched_skills": ["python", "flask", "mongodb"],
    "job_keywords": ["python", "flask", "react", "mongodb"]
  },
  "date_applied": "2023-10-30T14:22:18.763Z"
}
```

#### `GET /api/application-metrics/{application_id}`

Récupérer les métriques détaillées d'une candidature spécifique.

**Paramètres :**
- `application_id` : Identifiant de la candidature

**Réponse :**
```json
{
  "metrics": {
    "precision": 85.5,
    "recall": 75.0,
    "f1_score": 79.9,
    "accuracy": 82.3,
    "loss": 0.185,
    "application_id": "68017f6a8f8d0455b75f43dd",
    "date_applied": "2023-10-30T14:22:18.763Z",
    "skill_count": 15,
    "offer_keyword_count": 12
  },
  "matching_details": {
    "skill_match_score": 85.2,
    "semantic_similarity": 62.8,
    "final_score": 73.5
  }
}
```

### Génération de test technique

#### `POST /api/generate-test`

Générer un test technique basé sur des mots-clés de compétences.

**Paramètres (JSON) :**
- `keywords` ou `job_offer.keywords` (obligatoire) : Liste des compétences pour lesquelles générer un test
- `titre` ou `job_offer.titre` (optionnel) : Titre du test
- `niveau` ou `job_offer.niveau` (optionnel) : Niveau de difficulté (débutant, intermédiaire, avancé)

**Réponse :**
```json
{
  "test": {
    "title": "Test technique pour Développeur FullStack",
    "niveau": "intermédiaire",
    "qcm": [
      {
        "question": "Quelle méthode HTTP est utilisée pour récupérer des données d'une API REST?",
        "options": ["A: POST", "B: GET", "C: PUT", "D: DELETE"],
        "correct_answer": "B: GET",
        "explanation": "La méthode GET est utilisée pour récupérer des données sans les modifier."
      },
      {
        "question": "Comment définir une route dans Flask?",
        "options": [
          "A: @route('/path')",
          "B: @app.route('/path')",
          "C: route('/path')",
          "D: flask.route('/path')"
        ],
        "correct_answer": "B: @app.route('/path')",
        "explanation": "Flask utilise le décorateur @app.route pour définir les routes."
      }
    ],
    "keywords": ["python", "flask", "react"]
  },
  "job_offer": {
    "title": "Développeur FullStack",
    "keywords": ["python", "flask", "react", "mongodb"],
    "niveau": "intermédiaire"
  }
}
```

## Exemples d'utilisation

### Créer une offre d'emploi

```bash
curl -X POST http://localhost:5000/api/job-offers \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Développeur FullStack",
    "company": "TechSolutions",
    "description": "Nous recherchons un développeur full stack pour rejoindre notre équipe.",
    "location": "Paris",
    "level": "confirmé",
    "keywords": ["python", "flask", "react", "mongodb"]
  }'
```

### Postuler à une offre d'emploi

```bash
curl -X POST http://localhost:5000/api/apply \
  -F "cv=@mon_cv.pdf" \
  -F "offer_id=6800fa7847bc1978f0f9f974"
```

### Générer un test technique

```bash
curl -X POST http://localhost:5000/api/generate-test \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["python", "flask", "react"],
    "niveau": "intermédiaire"
  }'
```

## Pipeline CI/CD

Le projet utilise un pipeline CI/CD pour automatiser le déploiement et les tests:

1. **Build** : Construction de l'image Docker contenant l'application
2. **Test** : Exécution des tests unitaires et d'intégration
3. **Deploy** : Déploiement sur l'environnement cible (dev, staging, production)

### Exemple de fichier GitHub Actions

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 pytest
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
      - name: Test with pytest
        run: |
          pytest
      - name: Build Docker image
        run: |
          docker build -t cv-matching-app:latest .
```

## Notes additionnelles

- Les fichiers PDF sont traités avec PyPDF2
- Les fichiers DOCX sont traités avec docx2txt
- Les fichiers image sont traités avec pytesseract (OCR)
- L'API utilise SentenceTransformer pour l'analyse sémantique
- MongoDB est utilisé comme base de données
- Redis peut être utilisé pour la mise en cache (optionnel) 