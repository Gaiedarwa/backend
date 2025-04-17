# Application de recrutement

Une application Flask pour gérer les offres d'emploi et les candidatures avec analyse automatique des compétences et tests techniques.

## Fonctionnalités

- Création et gestion d'offres d'emploi
- Analyse de CV et comparaison avec les offres
- Génération de tests techniques adaptés aux compétences requises
- Stockage des données dans MongoDB
- Mise en cache avec Redis pour améliorer les performances

## Routes API

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | /job-offers | Créer une nouvelle offre d'emploi |
| GET | /job-offers | Récupérer toutes les offres d'emploi |
| GET | /job-offers/:id | Récupérer une offre d'emploi spécifique |
| POST | /apply | Postuler à une offre avec un CV |
| GET | /postulations | Récupérer toutes les candidatures |
| GET | /postulations/:id | Récupérer une candidature spécifique |
| POST | /generate_test | Générer un test technique |

## Installation

1. Cloner le dépôt

```bash
git clone https://github.com/username/recrutement-app.git
cd recrutement-app
```

2. Créer un environnement virtuel et l'activer

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

4. Configurer MongoDB et Redis

Créez un fichier `.env` à la racine du projet :

```
MONGO_URI=mongodb://localhost:27017/
REDIS_HOST=localhost
REDIS_PORT=6379
```

5. Lancer l'application

```bash
python app.py
```

L'application sera accessible à l'adresse http://localhost:5000

## Exemples d'utilisation

### Créer une offre d'emploi

```bash
curl -X POST http://localhost:5000/job-offers \
  -F "text=Nous recherchons un développeur Python senior avec expérience en Flask et MongoDB. Compétences requises : Python, Flask, REST API, MongoDB, Git."
```

### Postuler à une offre

```bash
curl -X POST http://localhost:5000/apply \
  -F "text_cv=Développeur Python avec 5 ans d'expérience en développement web. Maîtrise de Flask, Django, MongoDB et PostgreSQL." \
  -F "offer_id=1"
```

### Générer un test technique

```bash
curl -X POST http://localhost:5000/generate_test \
  -H "Content-Type: application/json" \
  -d '{"competencies":["python", "flask", "mongodb"], "difficulty":"Senior"}'
```

## Licence

MIT 