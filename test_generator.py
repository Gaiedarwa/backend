"""
Test generator module for creating automated tests based on job offers
Uses Ollama with llama3 for RAG-based test generation
"""
import requests
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API settings
OLLAMA_BASE_URL = "http://localhost:11434/api"  # Default Ollama API URL
OLLAMA_MODEL = "llama3"  # Default to llama3

def is_technical_domain(keywords, title=None, description=None):
    """
    Determine intelligently if the job is in a technical domain based on keywords and context
    
    Args:
        keywords (list): List of job keywords
        title (str, optional): Job title if available
        description (str, optional): Job description if available
        
    Returns:
        bool: True if technical domain, False otherwise
    """
    # Convertir tout en minuscules pour la comparaison
    keywords_lower = [k.lower() for k in keywords]
    
    # Domaines techniques courants et leurs mots-clés associés
    tech_domains = {
        "development": ["développement", "development", "code", "programming", "software", "logiciel"],
        "data": ["data", "données", "analytics", "analyse", "statistics", "statistiques"],
        "infrastructure": ["infrastructure", "devops", "cloud", "server", "serveur", "network", "réseau"],
        "security": ["security", "sécurité", "cyber", "cryptography", "cryptographie"],
        "ai": ["ai", "ia", "intelligence artificielle", "artificial intelligence", "machine learning", "deep learning"]
    }
    
    # Technologies spécifiques qui indiquent un poste technique
    specific_technologies = [
        # Langages de programmation
        "python", "java", "javascript", "c++", "c#", "ruby", "php", "go", "rust", "swift",
        # Frameworks et bibliothèques
        "django", "flask", "react", "angular", "vue", "node", "spring", "laravel", "rails",
        # Bases de données
        "sql", "nosql", "mysql", "postgresql", "mongodb", "oracle", "elasticsearch",
        # Outils de développement
        "git", "docker", "kubernetes", "jenkins", "terraform", "aws", "azure", "gcp"
    ]
    
    # 1. Vérifier si des technologies spécifiques sont mentionnées
    for tech in specific_technologies:
        if any(tech in keyword for keyword in keywords_lower):
            return True
    
    # 2. Vérifier s'il y a intersection avec des domaines techniques
    for domain_terms in tech_domains.values():
        if any(term in keyword for term in domain_terms for keyword in keywords_lower):
            return True
    
    # 3. Analyser le titre si disponible
    if title:
        title_lower = title.lower()
        technical_titles = ["developer", "développeur", "engineer", "ingénieur", "programmer", 
                           "programmeur", "architect", "architecte", "data scientist", 
                           "devops", "technicien", "technician", "admin", "administrator"]
        
        if any(tech_title in title_lower for tech_title in technical_titles):
            return True
    
    # 4. Faire une analyse contextuelle des mots-clés combinés
    # Compter le nombre de mots-clés qui semblent techniques
    technical_count = 0
    non_technical_count = 0
    
    non_technical_indicators = ["marketing", "sales", "vente", "finance", "accounting", "comptabilité", 
                               "hr", "human resources", "ressources humaines", "legal", "juridique",
                               "communication", "management", "gestion"]
    
    for keyword in keywords_lower:
        # Mots-clés indiquant clairement un domaine non technique
        if any(indicator in keyword for indicator in non_technical_indicators):
            non_technical_count += 1
        
        # Mots-clés qui pourraient indiquer un domaine technique
        if any(indicator in keyword for indicator in ["tech", "system", "système", "application", 
                                                     "platform", "plateforme", "digital", "numérique"]):
            technical_count += 1
    
    # Si plus d'indicateurs techniques que non techniques
    if technical_count > non_technical_count:
        return True
    
    # Par défaut, vérifier si au moins deux mots-clés techniques sont présents
    return technical_count >= 2

def determine_complexity(title):
    """
    Determine test complexity based on job title
    
    Args:
        title (str): Job title
        
    Returns:
        str: Complexity level (beginner, intermediate, or advanced)
    """
    title_lower = title.lower()
    
    # Define keywords for different levels
    senior_keywords = ['senior', 'lead', 'expert', 'architect', 'principal', 'chef']
    junior_keywords = ['junior', 'entry', 'débutant', 'stagiaire', 'intern']
    
    if any(keyword in title_lower for keyword in senior_keywords):
        return "advanced"
    elif any(keyword in title_lower for keyword in junior_keywords):
        return "beginner"
    else:
        return "intermediate"

def generate_prompt(job_offer):
    """
    Create a prompt for the LLM based on job offer data
    
    Args:
        job_offer (dict): Job offer data including title and keywords
        
    Returns:
        str: Formatted prompt for test generation
    """
    title = job_offer.get('title', '')
    keywords = job_offer.get('keywords', [])
    is_technical = is_technical_domain(keywords, title)
    complexity = determine_complexity(title)
    
    # Format keywords as a string
    keywords_str = ', '.join(keywords)
    
    # Create appropriate prompt based on domain
    if is_technical:
        prompt = f"""Tu es un concepteur de tests techniques spécialisé. Génère un test d'évaluation de compétences pour un poste de {title} avec une complexité {complexity}.
Le test doit couvrir les compétences suivantes: {keywords_str}.

Crée un test avec:
- 7 questions QCM sur les concepts théoriques
- 3 exercices de programmation pratiques

Pour chaque question QCM, inclus:
1. La question
2. 4 options de réponse (A, B, C, D)
3. La bonne réponse
4. Une brève explication pourquoi c'est la bonne réponse

Pour chaque exercice de programmation:
1. L'énoncé du problème
2. Un exemple d'entrée/sortie attendue
3. Des contraintes ou exigences spécifiques
4. Un squelette de code pour commencer
5. Une solution possible

Formate ta réponse en JSON avec cette structure:
{{
  "qcm": [
    {{
      "question": "...",
      "options": ["A: ...", "B: ...", "C: ...", "D: ..."],
      "correct_answer": "A/B/C/D",
      "explanation": "..."
    }}
  ],
  "programming_exercises": [
    {{
      "title": "...",
      "description": "...",
      "example": "...",
      "constraints": "...",
      "skeleton_code": "...",
      "solution": "..."
    }}
  ]
}}
"""
    else:
        prompt = f"""Tu es un concepteur de tests professionnels spécialisé. Génère un test d'évaluation de compétences pour un poste de {title} avec une complexité {complexity}.
Le test doit couvrir les compétences suivantes: {keywords_str}.

Crée un test avec 10 questions QCM qui évaluent efficacement ces compétences.

Pour chaque question, inclus:
1. La question
2. 4 options de réponse (A, B, C, D)
3. La bonne réponse
4. Une brève explication pourquoi c'est la bonne réponse

Formate ta réponse en JSON avec cette structure:
{{
  "qcm": [
    {{
      "question": "...",
      "options": ["A: ...", "B: ...", "C: ...", "D: ..."],
      "correct_answer": "A/B/C/D",
      "explanation": "..."
    }}
  ]
}}
"""
    
    return prompt

def query_ollama(prompt):
    """
    Send a query to Ollama API
    
    Args:
        prompt (str): The prompt to send to the model
        
    Returns:
        str: The generated response from Ollama
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120  # 2-minute timeout for generation
        )
        
        if response.status_code != 200:
            logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
            return None
        
        return response.json().get("response", "")
    
    except Exception as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        return None

def generate_test(job_offer):
    """
    Generate a test based on job offer data
    
    Args:
        job_offer (dict): Job offer data including title and keywords
        
    Returns:
        dict: Generated test data or error message
    """
    # Validate inputs
    if not job_offer.get('keywords'):
        return {"error": "Job offer keywords are required"}
    
    # Generate the prompt
    prompt = generate_prompt(job_offer)
    
    # Query the LLM
    response_text = query_ollama(prompt)
    if not response_text:
        return {"error": "Failed to generate test from LLM"}
    
    # Extract and parse JSON from the response
    try:
        # Try to find and extract JSON in case the LLM added extra text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            test_data = json.loads(json_text)
            
            # Validate the structure
            if "qcm" not in test_data:
                return {"error": "Generated test is missing QCM questions", "raw_response": response_text}
            
            # Add metadata
            test_data["metadata"] = {
                "job_title": job_offer.get('title', 'Non spécifié'),
                "keywords": job_offer.get('keywords', []),
                "is_technical": is_technical_domain(job_offer.get('keywords', []), job_offer.get('title', ''), job_offer.get('description', '')),
                "complexity": determine_complexity(job_offer.get('title', ''))
            }
            
            return test_data
            
        else:
            # If no JSON found, return the raw text for debugging
            return {"error": "Could not extract valid JSON from response", "raw_response": response_text}
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return {"error": "Failed to parse JSON from LLM response", "raw_response": response_text}