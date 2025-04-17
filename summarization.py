import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
import json
import os
from dotenv import load_dotenv

# Télécharger les ressources NLTK nécessaires lors de la première utilisation
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Charger les variables d'environnement
load_dotenv()

def summarize_concisely(text, max_length=200):
    """Résumer un texte en respectant une longueur maximale"""
    # Simplification: on prend les premières phrases jusqu'à max_length
    if len(text) <= max_length:
        return text
        
    sentences = re.split(r'(?<=[.!?])\s+', text)
    summary = ""
    
    for sentence in sentences:
        if len(summary) + len(sentence) + 1 <= max_length:
            summary += sentence + " "
        else:
            break
    
    return summary.strip()

def extract_keywords(text, max_keywords=10):
    """Extraire les mots-clés les plus importants d'un texte"""
    # Tokeniser le texte
    tokens = word_tokenize(text.lower())
    
    # Récupérer les stopwords
    stop_words = set(stopwords.words('french') + stopwords.words('english'))
    
    # Filtrer les stopwords et les tokens courts
    filtered_tokens = [token for token in tokens if token.isalpha() and len(token) > 3 and token not in stop_words]
    
    # Compter les occurrences
    word_count = Counter(filtered_tokens)
    
    # Récupérer les mots les plus fréquents
    most_common = word_count.most_common(max_keywords)
    
    return [word for word, count in most_common]

def generate_candidate_summary(skills, experience):
    """
    Génère un résumé structuré des compétences et expériences d'un candidat en utilisant LLaMA 3.2
    
    Args:
        skills (list): Liste des compétences extraites
        experience (list): Liste des expériences extraites
    
    Returns:
        str: Résumé organisé des compétences et expériences
    """
    
    # Tenter d'abord d'utiliser l'API LLaMA 3.2 si disponible
    api_key = os.getenv("LLAMA_API_KEY")
    api_url = os.getenv("LLAMA_API_URL")
    
    if api_key and api_url:
        try:
            prompt = f"""
            Tu es un assistant spécialisé dans la création de résumés professionnels.
            Génère un résumé structuré et concis (maximum 150 mots) pour un candidat 
            avec les compétences et expériences suivantes:
            
            Compétences: {', '.join(skills)}
            
            Expérience: {', '.join(experience)}
            
            N'invente aucune information. Utilise uniquement les données fournies.
            Le résumé doit être organisé de manière professionnelle, mettant en avant les points forts du profil.
            """
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.2",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            }
            
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API LLaMA: {str(e)}")
    
    # Méthode de repli si l'API n'est pas disponible
    skills_text = ", ".join(skills) if skills else "Non spécifié"
    exp_text = ", ".join(experience) if experience else "Non spécifié"
    
    summary = f"""
    Profil professionnel:
    
    Compétences: {skills_text}
    
    Expérience: {exp_text}
    
    Ce candidat possède des compétences en {skills_text[:100] + '...' if len(skills_text) > 100 else skills_text}
    avec une expérience en {exp_text[:100] + '...' if len(exp_text) > 100 else exp_text}.
    """
    
    return summary.strip()

def generate_job_description(keywords, niveau="Non spécifié"):
    """
    Génère une description d'offre d'emploi structurée à partir de mots-clés en utilisant LLaMA 3.2
    
    Args:
        keywords (list): Liste des mots-clés extraits
        niveau (str): Niveau requis pour le poste (Junior, Intermédiaire, Senior)
    
    Returns:
        str: Description structurée de l'offre d'emploi
    """
    
    # Tenter d'abord d'utiliser l'API LLaMA 3.2 si disponible
    api_key = os.getenv("LLAMA_API_KEY")
    api_url = os.getenv("LLAMA_API_URL")
    
    if api_key and api_url:
        try:
            prompt = f"""
            Tu es un assistant spécialisé dans la rédaction d'offres d'emploi.
            Génère une description de poste structurée et concise (maximum 200 mots) 
            basée sur les mots-clés suivants:
            
            Mots-clés: {', '.join(keywords)}
            Niveau: {niveau}
            
            N'invente aucune information supplémentaire. Utilise uniquement les mots-clés fournis.
            La description doit être professionnelle et inclure des sections comme:
            - Résumé du poste
            - Compétences requises
            - Expérience souhaitée
            """
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.2",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 400
            }
            
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API LLaMA: {str(e)}")
    
    # Méthode de repli si l'API n'est pas disponible
    keywords_text = ", ".join(keywords) if keywords else "Non spécifié"
    
    description = f"""
    Offre d'emploi: Professionnel(le) en {keywords_text[:50] + '...' if len(keywords_text) > 50 else keywords_text}
    
    Niveau: {niveau}
    
    Description du poste:
    Nous recherchons un(e) professionnel(le) pour rejoindre notre équipe.
    
    Compétences requises:
    {keywords_text}
    
    Expérience souhaitée:
    Selon le niveau {niveau}, une expérience appropriée sera demandée.
    """
    
    return description.strip() 