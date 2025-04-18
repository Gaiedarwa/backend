"""
Liste des termes techniques et technologies utilisés pour l'extraction de compétences.
Ce fichier centralise les définitions de technologies pour faciliter la maintenance.
"""

# Termes techniques communs au-delà de ce qui est dans TECH_SKILLS
ADDITIONAL_TECH_TERMS = [
    # Programming concepts
    "api", "rest", "soap", "microservices", "web services", "frontend", "backend",
    "full stack", "mvc", "orm", "sdk", "cli", "ui", "ux", "database", "cloud",
    
    # Methodologies
    "agile", "scrum", "kanban", "lean", "waterfall", "devops", "ci/cd", "tdd", "bdd",
    
    # Tools & platforms
    "jira", "confluence", "bitbucket", "aws", "azure", "gcp", "github", "gitlab",
    "jenkins", "docker", "kubernetes", "terraform", "ansible", "prometheus", "grafana",
    
    # Data & Analytics
    "analytics", "business intelligence", "big data", "data mining", "data science",
    "predictive modeling", "machine learning", "deep learning", "neural networks",
    "natural language processing", "nlp", "computer vision", 
    
    # Databases
    "sql", "nosql", "mysql", "postgresql", "mongodb", "cassandra", "redis", "elasticsearch",
    "neo4j", "oracle", "dynamodb",
    
    # Frameworks & libraries
    "spring", "hibernate", "django", "flask", "fastapi", "express", "react", "angular",
    "vue", "svelte", "jquery", "bootstrap", "tailwind", "tensorflow", "pytorch", "keras",
    
    # Security
    "cybersecurity", "encryption", "authentication", "authorization", "oauth", "jwt",
    "penetration testing", "security audit", "firewall", "vpn", "ssl", "tls"
]

# Termes spécifiques à différents domaines professionnels
DOMAIN_SPECIFIC_TERMS = {
    "development": [
        "développement", "programmation", "code", "développeur", "software engineer",
        "git", "version control", "api", "REST", "SOAP", "web services", "debugging"
    ],
    "data_science": [
        "data scientist", "statistiques", "R", "python", "machine learning", "deep learning",
        "neural networks", "data mining", "big data", "data analysis", "data visualization",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "regression", "classification"
    ],
    "devops": [
        "devops", "ci/cd", "continuous integration", "continuous deployment", "automation",
        "infrastructure as code", "docker", "kubernetes", "jenkins", "gitlab CI",
        "monitoring", "logging", "prometheus", "grafana", "terraform", "ansible"
    ],
    "security": [
        "cybersecurity", "security", "penetration testing", "vulnerability assessment",
        "security audit", "encryption", "cryptography", "authentication", "authorization",
        "firewall", "IDS", "IPS", "SIEM", "SOC", "threat intelligence", "OWASP"
    ],
    "design": [
        "ui", "ux", "user interface", "user experience", "wireframing", "prototyping",
        "figma", "sketch", "adobe xd", "photoshop", "illustrator", "responsive design",
        "accessibility", "usability testing", "information architecture"
    ],
    "management": [
        "project management", "product management", "agile", "scrum", "kanban",
        "leadership", "team management", "stakeholder management", "risk management",
        "resource planning", "roadmap", "sprint planning", "retrospective", "jira"
    ]
}

# Catégories de compétences critiques (utilisées pour la pondération des scores)
CRITICAL_SKILL_CATEGORIES = [
    "programming", "language", "framework", "database", "cloud", "devops", "security"
] 