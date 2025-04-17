from flask import Blueprint, request, jsonify
from bson.objectid import ObjectId
from services import process_document, remove_sensitive_data
from models import extract_entities
from summarization import extract_keywords, generate_candidate_summary, generate_job_description
from database import cv_collection, offers_collection
from datetime import datetime
import logging
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from pymongo import MongoClient
import redis
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from skills import TECH_SKILLS  # Import the TECH_SKILLS dictionary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask Blueprint
bp = Blueprint('routes', __name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client.cvs_db

# Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Sentence-Transformers model
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
sentence_model = SentenceTransformer(model_name)

# Utility functions for semantic matching
def get_embedding(text):
    if not text or text.strip() == "":
        return np.zeros(384)
    max_length = 5000
    if len(text) > max_length:
        text = text[:max_length]
    with torch.no_grad():
        embedding = sentence_model.encode(text)
    return embedding

def calculate_semantic_similarity(embedding1, embedding2):
    embedding1_reshaped = embedding1.reshape(1, -1)
    embedding2_reshaped = embedding2.reshape(1, -1)
    similarity = cosine_similarity(embedding1_reshaped, embedding2_reshaped)[0][0]
    return similarity

def extract_phone_numbers(text):
    """
    Enhanced phone number extraction with multiple pattern matching
    for various international formats.
    
    Args:
        text (str): The text to extract phone numbers from
        
    Returns:
        str: The first valid phone number found or None
    """
    if not text:
        return None
        
    # List of common phone number patterns in various formats
    patterns = [
        # International format with country code
        r'(?:\+|\(\+)\s*(?:\d{1,4})\s*(?:\)|\s|-|\.)*\s*\d{2,3}(?:\s|-|\.)*\d{2,3}(?:\s|-|\.)*\d{2,3}(?:\s|-|\.)*\d{2,3}',
        
        # Standard formats with various separators
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 123-456-7890
        r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b',  # 123-45-67-89
        
        # European/French formats
        r'\b0\d(?:[-.\s]?\d{2}){4}\b',  # 06 12 34 56 78 or 06-12-34-56-78
        r'\b\d{2}(?:[-.\s]?\d{2}){4}\b',  # 12 34 56 78 90
        
        # Simple digit sequences (as fallback)
        r'\b\d{10,12}\b',  # 10-12 digit number without separators
        
        # Format with parentheses
        r'\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}',  # (123) 456-7890
        
        # Format with country code in parentheses
        r'\(\+\d{1,3}\)[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # (+123) 456-789-0123
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Clean up the phone number (remove extra spaces, standardize format)
            phone = matches[0]
            # Remove non-digit characters except '+'
            phone = re.sub(r'[^\d+]', '', phone)
            
            # Format international numbers
            if phone.startswith('+'):
                return phone
            # Format numbers that start with 0 (add country code for France as an example)
            elif phone.startswith('0') and len(phone) >= 10:
                return '+33' + phone[1:]
            # Add + for numbers that appear to be international but missing the +
            elif len(phone) > 10 and not phone.startswith('0'):
                return '+' + phone
            # Default case
            else:
                return phone
    
    return None

def optimize_skill_extraction(text, job_related=False):
    """
    Enhanced skill extraction that works for both CVs and job offers.
    Identifies technical skills, tools, programming languages, and frameworks.
    
    Args:
        text (str): Text to extract skills from
        job_related (bool): Whether the text is from a job offer (affects weighting)
        
    Returns:
        list: List of extracted skills
    """
    # Base technical skills from our dictionary
    tech_skills = extract_skills_improved(text) if 'extract_skills_improved' in globals() else []
    
    # Common technical terms and keywords beyond what's in TECH_SKILLS
    additional_tech_terms = [
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
    
    found_skills = set(tech_skills)
    
    # Extract skills using regex patterns with word boundaries
    for skill in additional_tech_terms:
        # Create a pattern that matches the term as a whole word
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text.lower()):
            found_skills.add(skill)
    
    # Look for skill sections in the text
    skill_section = extract_skill_section(text)
    if skill_section:
        # Split by common separators and extract individual skills
        skill_items = re.split(r'[,;\n•·⋅◦▪▹➤➢✓✔→-]', skill_section)
        for item in skill_items:
            item = item.strip().lower()
            # Only consider items of reasonable length that look like technical terms
            if 2 < len(item) < 30 and not any(word in item for word in ["year", "ans", "je", "i am", "niveau", "level"]):
                found_skills.add(item)
    
    # Domain-specific skill extraction for job offers
    if job_related:
        # Extract skills that are mentioned as requirements
        requirement_patterns = [
            r'requis|required|mandatory|necessary|nécessaire',
            r'compétences|skills|expertise|proficiency',
            r'connaissances|knowledge|understanding',
            r'maîtrise|mastery|proficient in',
            r'expérience avec|experience with|familiar with'
        ]
        
        for pattern in requirement_patterns:
            matches = re.finditer(r'\b' + pattern + r'\b(.*?)(?:\.|;|\n)', text, re.IGNORECASE)
            for match in matches:
                req_text = match.group(1).lower()
                for skill in additional_tech_terms:
                    if skill.lower() in req_text:
                        found_skills.add(skill)
    
    # Normalize and clean skills
    cleaned_skills = []
    for skill in found_skills:
        # Convert to lowercase and strip whitespace
        skill = skill.lower().strip()
        if skill and 2 < len(skill) < 30:
            cleaned_skills.append(skill)
    
    return sorted(cleaned_skills)

def extract_skill_section(text):
    """
    Extract the skills section from a resume or job posting
    
    Args:
        text (str): The text to extract skills section from
        
    Returns:
        str: The extracted skills section or empty string if not found
    """
    # Common headers that could indicate a skills section
    skill_headers = [
        "compétences", "skills", "competencies", "technical skills", "technologies",
        "langages", "languages", "outils", "tools", "technical proficiency", 
        "compétences techniques", "expertise", "tech stack", "stack technique"
    ]
    
    # Try to find paragraphs that might contain skills
    paragraphs = re.split(r'\n\s*\n', text)
    skill_section = ""
    found_section = False
    
    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        
        # Check if this paragraph is a skills section header
        if any(header in para_lower for header in skill_headers):
            found_section = True
            
            # If this is the last paragraph, return an empty string
            if i == len(paragraphs) - 1:
                return ""
            
            # Take the next paragraph as the skills section
            skill_section = paragraphs[i+1]
            break
    
    if not found_section:
        # Try to find lines that indicate skills
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(header in line_lower for header in skill_headers) and i < len(lines) - 1:
                # Collect next few lines until empty line or new section
                for j in range(i+1, min(i+10, len(lines))):
                    if not lines[j].strip():
                        break
                    if any(re.match(rf"^{header}", lines[j].lower()) for header in ["experience", "education", "project", "formation"]):
                        break
                    skill_section += lines[j] + "\n"
                break
    
    return skill_section

def calculate_skill_match(cv_skills, job_keywords):
    """
    Calculate the match score between candidate skills and job keywords
    with enhanced accuracy and weighting
    
    Args:
        cv_skills (list): List of skills from the CV
        job_keywords (list): List of keywords from the job offer
        
    Returns:
        tuple: (match_score, matched_skills)
            - match_score: A score between 0-100 representing the match quality
            - matched_skills: List of skills that matched between CV and job offer
    """
    # If either list is empty, return minimum score
    if not cv_skills or not job_keywords:
        return 10, []
    
    # Convert to lowercase for case-insensitive matching
    cv_skills_lower = [skill.lower() for skill in cv_skills]
    job_keywords_lower = [kw.lower() for kw in job_keywords]
    
    # Find exact matches
    exact_matches = []
    for skill in cv_skills:
        skill_lower = skill.lower()
        # Check for exact match
        if skill_lower in job_keywords_lower:
            exact_matches.append(skill)
            continue
            
        # Check for substring match (e.g., "python" matches "python programming")
        for keyword in job_keywords_lower:
            if skill_lower in keyword or keyword in skill_lower:
                exact_matches.append(skill)
                break
    
    # Calculate direct match percentage
    exact_match_pct = (len(exact_matches) / len(job_keywords)) * 100 if job_keywords else 0
    
    # Find semantic matches
    semantic_matches = []
    semantic_scores = []
    
    for skill in cv_skills:
        if skill in exact_matches:
            continue  # Skip if already an exact match
            
        skill_embedding = get_embedding(skill.lower())
        best_match_score = 0
        best_match_keyword = None
        
        for keyword in job_keywords:
            keyword_embedding = get_embedding(keyword.lower())
            similarity = calculate_semantic_similarity(skill_embedding, keyword_embedding)
            
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_keyword = keyword
        
        # Only count strong semantic matches
        if best_match_score > 0.6:  # Threshold for semantic match
            semantic_matches.append(skill)
            semantic_scores.append(best_match_score)
    
    # Calculate semantic match percentage
    semantic_match_pct = 0
    if job_keywords and semantic_scores:
        semantic_match_pct = (sum(semantic_scores) / len(job_keywords)) * 100
    
    # Combine exact and semantic matches for the matched_skills list
    all_matches = exact_matches + semantic_matches
    
    # Calculate overall match score
    # Give higher weight to exact matches (70%) and lower to semantic matches (30%)
    match_score = (exact_match_pct * 0.7) + (semantic_match_pct * 0.3)
    
    # Apply bonus for having multiple matches
    match_bonus = min(40, len(all_matches) * 8)  # 8% per match up to 40%
    
    # Calculate final score
    final_score = match_score + match_bonus
    
    # Ensure score is between 10-100
    final_score = max(10, min(100, final_score))
    
    return round(final_score, 1), all_matches

# Define all routes on the Blueprint
@bp.route('/job-offers', methods=['POST'])
def create_job_offer():
    try:
        logger.info("Requête reçue sur /job-offers")
        data = request.json
        required_fields = ['title', 'company', 'description']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Le champ {field} est requis'}), 400
        description = generate_job_description(data['description'])
        job_offer = {
            'title': data['title'],
            'company': data['company'],
            'location': data.get('location', ''),
            'level': data.get('level', ''),
            'description': description['description'],
            'keywords': description['keywords'],
            'requirements': description['requirements'],
            'responsibilities': description['responsibilities']
        }
        job_id = offers_collection.insert_one(job_offer).inserted_id
        job_id = str(job_id)
        return jsonify({'job_id': job_id, 'job_offer': job_offer}), 201
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/job-offers', methods=['GET'])
def get_all_offers():
    try:
        offers = list(offers_collection.find())
        for offer in offers:
            offer["_id"] = str(offer["_id"])
        return jsonify(offers)
    except Exception as e:
        logging.error(f"Error in get_all_offers: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/job-offers/<offer_id>', methods=['GET'])
def get_offer_by_id(offer_id):
    try:
        offer = offers_collection.find_one({"_id": ObjectId(offer_id)})
        if not offer:
            return jsonify({"error": "Offre non trouvée"}), 404
        offer["_id"] = str(offer["_id"])
        return jsonify(offer)
    except Exception as e:
        logging.error(f"Error in get_offer_by_id: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/apply', methods=['POST'])
def apply_to_offer():
    try:
        logger.info("Requête reçue sur /apply")
        if 'cv' not in request.files or 'offer_id' not in request.form:
            return jsonify({'error': 'Le CV et l\'ID de l\'offre sont requis'}), 400
        cv_file = request.files['cv']
        offer_id = request.form['offer_id']
        if cv_file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        cv_text = ""
        filename = cv_file.filename
        if filename.endswith('.pdf'):
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(cv_file)
            for page in pdf_reader.pages:
                cv_text += page.extract_text() + "\n"
        elif filename.endswith('.docx'):
            import docx2txt
            cv_text = docx2txt.process(cv_file)
        elif filename.endswith('.txt'):
            cv_text = cv_file.read().decode('utf-8')
        else:
            return jsonify({'error': 'Format de fichier non supporté'}), 400
        
        job_offer = offers_collection.find_one({'_id': ObjectId(offer_id)})
        if not job_offer:
            return jsonify({'error': 'Offre d\'emploi non trouvée'}), 404
        
        entities = extract_entities(cv_text)
        
        # Améliorer la détection du numéro de téléphone
        phone = extract_phone_numbers(cv_text)
        if phone:
            entities['telephone'] = phone
        
        # Améliorer l'extraction des compétences
        extracted_skills = optimize_skill_extraction(cv_text)
        entities['skills'] = extracted_skills
        
        # Optimiser l'extraction des mots-clés de l'offre d'emploi
        job_description = job_offer.get('description', '')
        job_skills = optimize_skill_extraction(job_description, job_related=True)
        
        # Utiliser les mots-clés extraits ou ceux déjà présents dans l'offre
        job_keywords = job_offer.get('keywords', [])
        if job_skills and (not job_keywords or len(job_skills) > len(job_keywords)):
            job_keywords = job_skills
        
        # Supprimer les données sensibles du texte du CV
        sanitized_cv_text = remove_sensitive_data(cv_text)
        
        # Extraire les informations personnelles
        personal_info = {
            'name': entities.get('name', 'Nom non détecté'),
            'email': entities.get('email', 'Email non détecté'),
            'telephone': entities.get('telephone', 'Téléphone non détecté')
        }
        
        # Extraire les sections professionnelles pour le résumé
        sections = extract_professional_sections(sanitized_cv_text)
        
        # Générer un résumé optimisé à partir des sections professionnelles
        professional_content = sections.get('skills', '') + "\n\n" + sections.get('experience', '')
        professional_content = professional_content.strip()
        
        # Utiliser le contenu professionnel pour générer le résumé
        optimized_resume = generate_candidate_summary(entities.get('skills', []), entities.get('experience', []))
        
        # Calculer la correspondance des compétences et la similarité sémantique avec l'algorithme amélioré
        skill_match_score, matched_skills = calculate_skill_match(entities.get('skills', []), job_keywords)
        
        # Calculer similarité sémantique entre CV et description du poste
        semantic_similarity = calculate_semantic_similarity(
            get_embedding(sanitized_cv_text),
            get_embedding(job_offer.get('description', ''))
        ) * 100
        
        # Calculer score final (combinaison pondérée)
        final_score = (skill_match_score * 0.7) + (semantic_similarity * 0.3)
        final_score = max(10, min(100, final_score))
        
        application_id = cv_collection.insert_one({
            'offer_id': offer_id,
            'cv_text': sanitized_cv_text,
            'entities': entities,
            'personal_info': personal_info,
            'optimized_resume': optimized_resume,
            'match_data': {
                'skill_match_score': skill_match_score,
                'semantic_similarity': round(semantic_similarity, 1),
                'final_score': round(final_score, 1),
                'matched_skills': matched_skills,
                'job_keywords': job_keywords
            },
            'date_applied': datetime.now()
        }).inserted_id
        
        application_id = str(application_id)
        
        response = {
            'application_id': application_id,
            'candidate': {
                'personal_info': personal_info,
                'optimized_resume': optimized_resume,
                'skills': entities.get('skills', [])
            },
            'job_offer': {
                'id': str(job_offer['_id']),
                'title': job_offer.get('title', 'Non spécifié'),
                'company': job_offer.get('company', 'Non spécifié'),
                'keywords': job_keywords
            },
            'matching': {
                'score': round(final_score, 1),
                'skill_match_score': skill_match_score,
                'semantic_similarity': round(semantic_similarity, 1),
                'matched_skills': matched_skills
            }
        }
        logger.info(f"Candidature traitée avec succès - Matching score: {final_score}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/postulations', methods=['GET'])
def get_all_postulations():
    try:
        postulations = list(cv_collection.find())
        for postulation in postulations:
            postulation["_id"] = str(postulation["_id"])
        return jsonify(postulations)
    except Exception as e:
        logging.error(f"Error in get_all_postulations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/postulations/<postulation_id>', methods=['GET'])
def get_postulation_by_id(postulation_id):
    try:
        postulation = cv_collection.find_one({"_id": ObjectId(postulation_id)})
        if not postulation:
            return jsonify({"error": "Postulation non trouvée"}), 404
        postulation["_id"] = str(postulation["_id"])
        return jsonify(postulation)
    except Exception as e:
        logging.error(f"Error in get_postulation_by_id: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_professional_sections(cv_text):
    """
    Extraire les sections professionnelles d'un CV (compétences, expérience, etc.)
    en excluant les informations personnelles.
    
    Args:
        cv_text (str): Texte du CV
        
    Returns:
        dict: Sections professionnelles extraites
    """
    sections = {
        'skills': '',
        'experience': '',
        'education': '',
        'projects': ''
    }
    
    # Supprimer les informations personnelles (email, téléphone, adresse)
    # Patterns pour les informations de contact
    contact_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Téléphone international
        r'\b\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b',  # Téléphone français
        r'\b\d+\s+[A-Za-z\s]+,\s+[A-Za-z\s]+,\s+[A-Za-z]{2}\s+\d{5}\b',  # Adresse
        r'https?://\S+',  # URLs
        r'www\.\S+'  # Web links
    ]
    
    # Remplacer les informations de contact par des placeholders
    sanitized_text = cv_text
    for pattern in contact_patterns:
        sanitized_text = re.sub(pattern, '[INFORMATION PERSONNELLE]', sanitized_text)
    
    # Diviser le texte en sections basées sur des marqueurs courants
    sections_markers = {
        'skills': [
            r'(?:compétences|skills|technical skills|competencies|expertise)',
            r'(?:langages|languages|outils|tools|technologies)'
        ],
        'experience': [
            r'(?:expérience|experience|work experience|employment|parcours professionnel)',
            r'(?:career|carrière|postes|positions)'
        ],
        'education': [
            r'(?:éducation|education|formation|studies|parcours académique|diplômes|degrees)',
            r'(?:qualifications|certifications|academic)'
        ],
        'projects': [
            r'(?:projets|projects|réalisations|achievements|portfolio)',
            r'(?:works|travaux|développements|developments)'
        ]
    }
    
    # Trouver les sections pertinentes
    paragraphs = re.split(r'\n\s*\n', sanitized_text)
    current_section = None
    
    for para in paragraphs:
        para_lower = para.lower()
        
        # Déterminer la section actuelle
        for section, markers in sections_markers.items():
            if any(re.search(marker, para_lower, re.IGNORECASE) for marker in markers):
                current_section = section
                break
        
        # Ajouter le contenu à la section appropriée
        if current_section and len(para) > 10:  # Éviter les paragraphes trop courts
            sections[current_section] += para + "\n\n"
    
    return sections

# Initialize Flask routes
def init_routes(app):
    # Register the Blueprint with the app
    app.register_blueprint(bp, url_prefix='/api')