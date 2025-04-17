import random

def generate_tests(keywords, niveau):
    """Générer une liste de questions techniques basées sur des mots-clés et un niveau"""
    tests = []
    
    # Questions sur Python
    python_questions = {
        "Junior": [
            "Quelle est la différence entre une liste et un tuple en Python ?",
            "Comment créer une fonction en Python ?",
            "Expliquez ce qu'est une compréhension de liste.",
            "Comment gérer les exceptions en Python ?"
        ],
        "Intermédiaire": [
            "Qu'est-ce qu'un décorateur en Python et comment l'implémenter ?",
            "Expliquez le concept de générateurs en Python.",
            "Comment fonctionne l'héritage multiple en Python ?",
            "Quelle est la différence entre `__str__` et `__repr__` ?"
        ],
        "Senior": [
            "Expliquez les métaclasses en Python et donnez un exemple d'utilisation.",
            "Comment optimiser les performances d'un script Python lent ?",
            "Expliquez le GIL (Global Interpreter Lock) et ses implications.",
            "Comment implémenter un context manager personnalisé ?"
        ]
    }
    
    # Questions sur JavaScript
    javascript_questions = {
        "Junior": [
            "Quelle est la différence entre `let`, `const` et `var` ?",
            "Expliquez le concept de fermeture (closure) en JavaScript.",
            "Comment ajouter un événement à un élément HTML ?",
            "Qu'est-ce que le DOM ?"
        ],
        "Intermédiaire": [
            "Expliquez les Promesses en JavaScript et leur utilité.",
            "Quelle est la différence entre `==` et `===` ?",
            "Comment fonctionne l'héritage prototypal en JavaScript ?",
            "Expliquez le concept de hoisting."
        ],
        "Senior": [
            "Expliquez le modèle d'événement en JavaScript et la propagation des événements.",
            "Comment optimiser les performances d'une application JavaScript ?",
            "Expliquez le concept de Web Workers et leur utilisation.",
            "Implémentez une fonction de debounce."
        ]
    }
    
    # Questions sur SQL
    sql_questions = {
        "Junior": [
            "Quelle est la différence entre DELETE et TRUNCATE ?",
            "Expliquez ce qu'est une clé primaire et une clé étrangère.",
            "Comment effectuer une jointure simple entre deux tables ?",
            "Qu'est-ce qu'une requête SELECT ?"
        ],
        "Intermédiaire": [
            "Expliquez les différents types de jointures en SQL.",
            "Comment optimiser une requête SQL lente ?",
            "Qu'est-ce qu'un index et comment l'utiliser ?",
            "Expliquez les transactions SQL."
        ],
        "Senior": [
            "Comment concevoir un schéma de base de données pour une application à haute disponibilité ?",
            "Expliquez les verrous de table et de ligne en SQL.",
            "Comment écrire des requêtes SQL complexes avec des sous-requêtes ?",
            "Expliquez les techniques d'optimisation avancées en SQL."
        ]
    }
    
    # Questions sur React
    react_questions = {
        "Junior": [
            "Qu'est-ce qu'un composant React ?",
            "Quelle est la différence entre state et props ?",
            "Expliquez le cycle de vie d'un composant React.",
            "Comment créer un formulaire contrôlé en React ?"
        ],
        "Intermédiaire": [
            "Expliquez les hooks React et leur utilité.",
            "Comment gérer les effets secondaires avec useEffect ?",
            "Quelle est la différence entre un composant fonctionnel et un composant de classe ?",
            "Expliquez le concept de Context API."
        ],
        "Senior": [
            "Comment optimiser les performances d'une application React ?",
            "Expliquez le rendu côté serveur (SSR) avec React.",
            "Comment implémenter un système de gestion d'état global sans Redux ?",
            "Expliquez les techniques avancées de mémoïsation en React."
        ]
    }
    
    # Correspondance des mots-clés avec les catégories de questions
    question_categories = {
        "python": python_questions,
        "javascript": javascript_questions,
        "sql": sql_questions,
        "react": react_questions
    }
    
    # Si le niveau n'est pas spécifié, utiliser un niveau par défaut
    if niveau not in ["Junior", "Intermédiaire", "Senior"]:
        niveau = "Intermédiaire"
    
    # Parcourir les mots-clés et ajouter des questions pertinentes
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for category, questions in question_categories.items():
            if category in keyword_lower:
                # Ajouter des questions de cette catégorie
                category_questions = questions.get(niveau, questions["Intermédiaire"])
                tests.extend(category_questions)
                break
    
    # Si pas assez de questions trouvées, ajouter des questions générales
    general_questions = {
        "Junior": [
            "Qu'est-ce que l'intégration continue / déploiement continu (CI/CD) ?",
            "Expliquez le concept de base de contrôle de version.",
            "Quelle est la différence entre une API REST et une API SOAP ?",
            "Qu'est-ce que le débogage ?"
        ],
        "Intermédiaire": [
            "Expliquez le concept de microservices.",
            "Comment sécuriser une API web ?",
            "Qu'est-ce que le test unitaire et pourquoi est-il important ?",
            "Expliquez le concept de conteneurisation."
        ],
        "Senior": [
            "Comment concevoir une architecture scalable pour une application web ?",
            "Expliquez les principes SOLID de la programmation orientée objet.",
            "Comment gérer la sécurité et la confidentialité des données dans une application ?",
            "Expliquez les différentes stratégies de déploiement (blue-green, canary, etc.)."
        ]
    }
    
    if len(tests) < 5:
        tests.extend(general_questions[niveau])
    
    # Supprimer les doublons et mélanger les questions
    tests = list(set(tests))
    random.shuffle(tests)
    
    return tests[:20]  # Limiter à 20 questions maximum

def select_random_test(tests, count=1):
    """Sélectionner un nombre spécifique de tests au hasard"""
    if count >= len(tests):
        return tests
    return random.sample(tests, count)

def get_user_input(prompt):
    """Obtenir une entrée utilisateur avec un prompt"""
    return input(prompt) 