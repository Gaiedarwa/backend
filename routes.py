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
from metrics import calculate_matching_metrics  # Import the metrics module
from test_generator import generate_test  # Import the test generator module
from open_application import OpenApplicationProcessor  # Import the open application processor
import ollama

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

# Initialize the OpenApplicationProcessor
open_application_processor = OpenApplicationProcessor(cv_collection, offers_collection, sentence_model)

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
    if not text:
        return []
        
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
            # Correction ici pour éviter l'erreur quand match.group(1) est None
            matches = re.finditer(r'\b' + pattern + r'\b(.*?)(?:\.|;|\n|$)', text, re.IGNORECASE)
            for match in matches:
                # Vérifier si le groupe est None avant d'appeler lower()
                if match.group(1) is not None:
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

def calculate_matching_metrics(candidate_skills, job_requirements, matched_skills):
    """
    Calculate precision, recall, F1-score and accuracy for skill matching
    
    Args:
        candidate_skills (list): List of skills from the candidate's CV
        job_requirements (list): List of required skills from the job description
        matched_skills (list): List of skills that matched between CV and job
        
    Returns:
        dict: Dictionary containing various matching metrics
    """
    if not candidate_skills or not job_requirements:
        return {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "accuracy": 0,
            "loss": 1.0
        }
    
    # Normalize all to lowercase for comparison
    candidate_skills_lower = [s.lower() for s in candidate_skills]
    job_requirements_lower = [s.lower() for s in job_requirements]
    matched_skills_lower = [s.lower() for s in matched_skills]
    
    # Calculate true positives, false positives, and false negatives
    true_positives = len(matched_skills_lower)  # Skills that matched
    
    # False positives: skills incorrectly matched (shouldn't happen in our case, but kept for completeness)
    false_positives = sum(1 for skill in matched_skills_lower if skill not in job_requirements_lower)
    
    # False negatives: required skills missing from matches
    false_negatives = sum(1 for skill in job_requirements_lower if skill not in matched_skills_lower)
    
    # True negatives: skills correctly not matched (skills not in CV and not required by job)
    # This is harder to define precisely, but we can approximate:
    all_unique_skills = set(candidate_skills_lower).union(set(job_requirements_lower))
    true_negatives = len(all_unique_skills) - (true_positives + false_positives + false_negatives)
    
    # Calculate precision: what percentage of matched skills were actually required
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Calculate recall: what percentage of required skills were matched
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score: harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy: percentage of correct predictions (true positives + true negatives)
    total = true_positives + false_positives + false_negatives + true_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    # Calculate loss using cross-entropy inspired approach
    # Higher loss when important skills are missing
    if job_requirements_lower:
        matched_ratio = len(matched_skills_lower) / len(job_requirements_lower)
        # Loss is higher when fewer job requirements are matched
        loss = 1.0 - matched_ratio
    else:
        loss = 0.0
        
    # Apply weighting to prioritize matching critical skills
    critical_skill_count = 0
    matched_critical_count = 0
    
    # Some skills might be more important than others (e.g., programming languages)
    critical_categories = ["programming", "language", "framework", "database", "cloud"]
    
    for skill in job_requirements_lower:
        if any(category in skill for category in critical_categories):
            critical_skill_count += 1
            if skill in matched_skills_lower:
                matched_critical_count += 1
    
    # If there are critical skills, adjust metrics accordingly
    if critical_skill_count > 0:
        critical_recall = matched_critical_count / critical_skill_count
        # Blend regular recall with critical recall (60% weight to critical skills)
        adjusted_recall = (recall * 0.4) + (critical_recall * 0.6)
        
        # Adjust loss to penalize missing critical skills more heavily
        critical_loss = 1.0 - critical_recall
        loss = (loss * 0.4) + (critical_loss * 0.6)
        
        # Update other metrics
        if precision > 0:
            f1_score = 2 * (precision * adjusted_recall) / (precision + adjusted_recall)
    
    # Ensure all values are between 0 and 1, then convert to percentages
    metrics = {
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1_score": round(f1_score * 100, 1),
        "accuracy": round(accuracy * 100, 1),
        "loss": round(loss, 3)
    }
    
    return metrics

@bp.route('/apply', methods=['POST'])
def apply_to_offer():
    try:
        logger.info("Requête reçue sur /apply")
        
        # Vérifier la présence des éléments requis
        if 'cv' not in request.files or 'offer_id' not in request.form:
            return jsonify({'error': 'Le CV et l\'ID de l\'offre sont requis'}), 400
        
        cv_file = request.files['cv']
        offer_id = request.form['offer_id']
        if cv_file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Récupérer les champs de formulaire supplémentaires (optionnels)
        form_name = request.form.get('name', '')
        form_email = request.form.get('email', '')
        form_telephone = request.form.get('telephone', '')
        
        # Extraire le texte du CV
        cv_text = ""
        filename = cv_file.filename.lower()  # Normaliser l'extension en minuscules
        
        try:
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
            elif any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                # OCR pour les images
                try:
                    import pytesseract
                    from PIL import Image
                    import io
                    
                    image = Image.open(io.BytesIO(cv_file.read()))
                    cv_text = pytesseract.image_to_string(image, lang='fra+eng')
                except ImportError:
                    return jsonify({'error': 'Bibliothèques pour le traitement d\'image non installées. Essayez: pip install pytesseract Pillow'}), 500
            else:
                return jsonify({'error': 'Format de fichier non supporté. Utilisez PDF, DOCX, TXT ou JPG/PNG'}), 400
        except Exception as file_error:
            logger.error(f"Erreur lors de l'extraction du texte du CV: {str(file_error)}")
            return jsonify({'error': f"Erreur lors de l'extraction du texte du CV: {str(file_error)}"}), 400
        
        # Vérifier si le texte a été correctement extrait
        if not cv_text or len(cv_text.strip()) < 50:
            return jsonify({'error': 'Impossible d\'extraire suffisamment de texte du CV. Vérifiez que le fichier n\'est pas corrompu ou vide.'}), 400
        
        # Récupérer l'offre d'emploi
        try:
            job_offer = offers_collection.find_one({'_id': ObjectId(offer_id)})
            if not job_offer:
                return jsonify({'error': 'Offre d\'emploi non trouvée'}), 404
        except Exception as db_error:
            logger.error(f"Erreur lors de la récupération de l'offre: {str(db_error)}")
            return jsonify({'error': f"Erreur lors de la récupération de l'offre: {str(db_error)}"}), 400
        
        # Extraire les entités et les compétences
        entities = extract_entities(cv_text)
        
        # Utiliser les informations du formulaire si fournies, sinon utiliser les informations extraites
        if form_name:
            entities['name'] = form_name
        else:
            # Améliorer la détection du nom avec la nouvelle fonction
            detected_name = extract_name_improved(cv_text)
            if detected_name:
                entities['name'] = detected_name
        
        if form_email:
            entities['email'] = form_email
                
        if form_telephone:
            entities['telephone'] = form_telephone
        else:
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
        try:
            optimized_resume = generate_candidate_summary(entities.get('skills', []), entities.get('experience', []))
        except Exception as resume_error:
            logger.error(f"Erreur lors de la génération du résumé: {str(resume_error)}")
            optimized_resume = "Résumé non disponible en raison d'une erreur de traitement."
        
        # Calculer la correspondance des compétences et la similarité sémantique avec l'algorithme amélioré
        try:
            skill_match_score, matched_skills = calculate_skill_match(entities.get('skills', []), job_keywords)
            
            # Calculer similarité sémantique avec la fonction améliorée
            semantic_similarity = calculate_semantic_similarity_enhanced(
                sanitized_cv_text,
                job_offer.get('description', '')
            ) * 100
            
            # Calculer score final (combinaison pondérée)
            final_score = (skill_match_score * 0.7) + (semantic_similarity * 0.3)
            final_score = max(10, min(100, final_score))
            
            # Calculer des métriques détaillées, mais ne pas les inclure dans la réponse principale
            metrics = calculate_matching_metrics(entities.get('skills', []), job_keywords, matched_skills)
        except Exception as match_error:
            logger.error(f"Erreur lors du calcul des scores de matching: {str(match_error)}")
            skill_match_score = 50
            semantic_similarity = 50
            final_score = 50
            matched_skills = []
            metrics = {
                "precision": 0,
                "recall": 0, 
                "f1_score": 0,
                "accuracy": 0,
                "loss": 1.0
            }
        
        # Enregistrer la candidature dans la base de données
        try:
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
                    'job_keywords': job_keywords,
                    'metrics': metrics  # Conserver les métriques dans la base de données
                },
                'date_applied': datetime.now(),
                'source': 'form_submission'
            }).inserted_id
            
            application_id = str(application_id)
        except Exception as db_error:
            logger.error(f"Erreur lors de l'enregistrement en base de données: {str(db_error)}")
            return jsonify({'error': f"Erreur lors de l'enregistrement de la candidature: {str(db_error)}"}), 500
        
        # Préparer la réponse sans inclure les métriques
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
                # Métriques détaillées supprimées de la réponse principale
            }
        }
        
        logger.info(f"Candidature traitée avec succès - ID: {application_id}, Score: {final_score}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/job-offers', methods=['POST'])
def job_offers_post():
    
    try:
        logger.info("Requête reçue sur /api/job-offers (POST)")
        
        # Vérifier d'abord si on a des données JSON
        if request.is_json:
            data = request.json
            if not data:
                return jsonify({'error': 'Données JSON requises ou fichier d\'offre d\'emploi'}), 400
            
            required_fields = ['title', 'company']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Le champ {field} est requis'}), 400
            
            title = data['title']
            company = data['company']
            location = data.get('location', '')
            level = data.get('level', '')
            description_text = data.get('description', '')
            job_id = data.get('job_id', '')
            work_type = data.get('work_type', '')
            contract_type = data.get('contract_type', '')
            responsibilities = data.get('responsibilities', [])
            profile = data.get('profile', [])
            
            # Récupérer les mots-clés s'ils sont fournis
            if 'keywords' in data:
                if isinstance(data['keywords'], list):
                    keywords = data['keywords']
                else:
                    keywords = [k.strip() for k in data['keywords'].split(',')]
            else:
                # Extraire les mots-clés du texte
                keywords = optimize_skill_extraction(description_text, job_related=True)
            
            # Déterminer automatiquement le niveau si non fourni
            if not level:
                level = determine_job_level(description_text, profile, title)
            
            # Générer une description enrichie si nécessaire
            if description_text:
                desc_result = generate_job_description(keywords)
                
                # Vérifier si le résultat est un dictionnaire
                if isinstance(desc_result, dict):
                    description = desc_result.get('description', description_text)
                    # N'écrasez les mots-clés que s'ils n'ont pas été fournis
                    if not keywords:
                        keywords = desc_result.get('keywords', [])
                    if not responsibilities:
                        responsibilities = desc_result.get('responsibilities', [])
                else:
                    description = desc_result if desc_result else description_text
                    if not keywords:
                        keywords = optimize_skill_extraction(description_text, job_related=True)
            else:
                description = ""
            
            # Création de l'objet job_offer
            job_offer = {
                'job_id': job_id,
                'title': title,
                'company': company,
                'location': location,
                'level': level,
                'description': description,
                'work_type': work_type, 
                'contract_type': contract_type,
                'keywords': keywords,
                'responsibilities': responsibilities,
                'profile': profile,
                'created_at': datetime.now()
            }
            
            # Insertion en base de données
            job_id = offers_collection.insert_one(job_offer).inserted_id
            job_offer['_id'] = str(job_id)
            
            return jsonify({'job_id': str(job_id), 'job_offer': job_offer, 'extracted_keywords': keywords}), 201
        
        # Si pas de JSON, rechercher les données dans form data et fichiers
        job_file = None
        
        # Vérifier tous les noms possibles du fichier
        if 'job_document' in request.files and request.files['job_document'].filename:
            job_file = request.files['job_document']
        elif 'offer' in request.files and request.files['offer'].filename:
            job_file = request.files['offer']
        
        job_text = ""
        
        # Extraire le texte du fichier si disponible
        if job_file:
            filename = job_file.filename.lower()
            
            try:
                if filename.endswith('.pdf'):
                    # Extraction de texte du PDF
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(job_file)
                    for page in pdf_reader.pages:
                        job_text += page.extract_text() + "\n"
                elif any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    # OCR pour les images
                    try:
                        import pytesseract
                        from PIL import Image
                        import io
                        
                        image = Image.open(io.BytesIO(job_file.read()))
                        job_text = pytesseract.image_to_string(image, lang='fra+eng')
                    except ImportError:
                        return jsonify({'error': 'Libraries for image processing not installed. Try: pip install pytesseract Pillow'}), 500
                else:
                    return jsonify({'error': 'Format de fichier non supporté. Utilisez PDF, JPG ou PNG'}), 400
            except Exception as file_error:
                logger.error(f"Erreur lors de l'extraction du texte du fichier: {str(file_error)}")
                return jsonify({'error': f"Erreur lors de l'extraction du texte du fichier: {str(file_error)}"}), 400
        
        # Récupérer texte supplémentaire (optionnel)
        additional_text = request.form.get('job_text', '')
        if additional_text:
            job_text = job_text + "\n\n" + additional_text
        
        # Récupérer les champs du formulaire
        title = request.form.get('title', 'Poste non spécifié')
        company = request.form.get('company', 'Entreprise non spécifiée')
        location = request.form.get('location', '')
        level = request.form.get('level', '')
        custom_job_id = request.form.get('job_id', '')
        work_type = request.form.get('work_type', '')
        contract_type = request.form.get('contract_type', '')
        
        # Récupérer les sections spécifiques si fournies dans le formulaire
        responsibilities_text = request.form.get('responsibilities', '')
        responsibilities = [r.strip() for r in responsibilities_text.split('\n')] if responsibilities_text else []
        
        profile_text = request.form.get('profile', '')
        profile = [p.strip() for p in profile_text.split('\n')] if profile_text else []
        
        # Récupérer les mots-clés s'ils sont fournis
        keywords_text = request.form.get('keywords', '')
        if keywords_text:
            keywords = [k.strip() for k in keywords_text.split(',')]
        elif job_text:
            keywords = optimize_skill_extraction(job_text, job_related=True)
        else:
            keywords = []
        
        # Déterminer automatiquement le niveau si non fourni
        if not level:
            level = determine_job_level(job_text, profile, title)
        
        # Générer une description enrichie si possible
        enriched_description = ""
        if not responsibilities and keywords:
            try:
                desc_result = generate_job_description(keywords)
                
                # Si la génération a retourné un dictionnaire, utiliser ses valeurs
                if isinstance(desc_result, dict):
                    enriched_description = desc_result.get('description', '')
                    if not responsibilities:
                        responsibilities = desc_result.get('responsibilities', [])
                    if not profile:
                        profile = desc_result.get('requirements', [])
                else:
                    enriched_description = desc_result if desc_result else job_text
            except Exception as gen_error:
                logger.error(f"Erreur lors de la génération de la description: {str(gen_error)}")
        
        # Construire l'objet d'offre d'emploi
        job_offer = {
            'job_id': custom_job_id,
            'title': title,
            'company': company,
            'location': location,
            'level': level,
            'work_type': work_type,
            'contract_type': contract_type,
            'description': job_text,  # Texte brut de l'offre (peut être vide)
            'enriched_description': enriched_description,  # Description enrichie par l'IA (peut être vide)
            'keywords': keywords,
            'responsibilities': responsibilities,
            'profile': profile,
            'created_at': datetime.now(),
            'source': 'file_upload' if job_file else 'form_input'
        }
        
        # Insertion en base de données
        job_id = offers_collection.insert_one(job_offer).inserted_id
        job_offer['_id'] = str(job_id)
        
        return jsonify({
            'job_id': str(job_id), 
            'job_offer': job_offer, 
            'extracted_keywords': keywords,
            'level': level
        }), 201
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def determine_job_level(description_text, profile, title):
    """
    Détermine automatiquement le niveau de l'offre d'emploi (junior/senior)
    en fonction du texte de description, du profil requis et du titre.
    
    Args:
        description_text (str): Texte de description de l'offre
        profile (list): Liste des exigences du profil
        title (str): Titre du poste
        
    Returns:
        str: 'junior', 'intermédiaire' ou 'senior'
    """
    # Convertir en texte pour l'analyse
    text_to_analyze = ""
    if description_text:
        text_to_analyze += description_text + " "
    
    if isinstance(profile, list):
        text_to_analyze += " ".join(profile) + " "
    elif isinstance(profile, str):
        text_to_analyze += profile + " "
        
    text_to_analyze += title
    text_to_analyze = text_to_analyze.lower()
    
    # Termes indiquant un poste senior
    senior_terms = [
        'senior', 'lead', 'principal', 'chef', 'architecte', 'manager', 'director',
        'expert', 'avancé', 'advanced', 'expérimenté', 'experienced',
        '5+ ans', '5+ years', '7+ ans', '7+ years', '10+ ans', '10+ years',
        'responsable', 'head of', 'strategy', 'stratégie', 'leadership'
    ]
    
    # Termes indiquant un poste junior
    junior_terms = [
        'junior', 'stagiaire', 'intern', 'débutant', 'beginner', 'entry',
        'apprenti', 'trainee', 'assistant', 'étudiant', 'student',
        'niveau 1', 'level 1', '0-2 ans', '0-2 years', '1-3 ans', '1-3 years'
    ]
    
    # Termes indiquant un niveau intermédiaire
    intermediate_terms = [
        'intermédiaire', 'intermediate', 'confirmé', 'confirmed', 'mid-level',
        '2-5 ans', '2-5 years', '3-5 ans', '3-5 years'
    ]
    
    # Compter les occurrences des termes
    senior_count = sum(1 for term in senior_terms if term in text_to_analyze)
    junior_count = sum(1 for term in junior_terms if term in text_to_analyze)
    intermediate_count = sum(1 for term in intermediate_terms if term in text_to_analyze)
    
    # Déterminer le niveau en fonction du nombre d'occurrences
    if senior_count > junior_count and senior_count > intermediate_count:
        return 'senior'
    elif junior_count > senior_count and junior_count > intermediate_count:
        return 'junior'
    elif intermediate_count > senior_count and intermediate_count > junior_count:
        return 'intermédiaire'
    elif senior_count > 0:
        return 'senior'  # Par défaut senior si quelques termes senior trouvés
    elif junior_count > 0:
        return 'junior'  # Par défaut junior si quelques termes junior trouvés
    else:
        return 'intermédiaire'  # Niveau par défaut

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

@bp.route('/apply-open', methods=['POST'])
def apply_open():
    """
    Route pour les candidatures libres sans ID d'offre spécifique.
    Le système trouve automatiquement les offres d'emploi les plus adaptées au CV.
    """
    try:
        logger.info("Requête reçue sur /apply-open")
        if 'cv' not in request.files:
            return jsonify({'error': 'Le CV est requis'}), 400
        
        cv_file = request.files['cv']
        if cv_file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Extraire le texte du CV
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
        
        # Extraire les entités et compétences
        entities = extract_entities(cv_text)
        phone = extract_phone_numbers(cv_text)
        if phone:
            entities['telephone'] = phone
        
        extracted_skills = optimize_skill_extraction(cv_text)
        entities['skills'] = extracted_skills
        
        # Nettoyer les données sensibles du CV
        sanitized_cv_text = remove_sensitive_data(cv_text)
        
        # Extraire les informations personnelles
        personal_info = {
            'name': entities.get('name', 'Nom non détecté'),
            'email': entities.get('email', 'Email non détecté'),
            'telephone': entities.get('telephone', 'Téléphone non détecté')
        }
        
        # Générer un résumé optimisé
        optimized_resume = generate_candidate_summary(entities.get('skills', []), entities.get('experience', []))
        
        # Traiter la candidature libre
        response, status_code = open_application_processor.process_open_application(
            cv_text, 
            entities, 
            sanitized_cv_text, 
            personal_info, 
            optimized_resume,
            calculate_matching_metrics
        )
        
        if status_code == 200:
            logger.info(f"Candidature libre traitée avec succès - ID: {response['application_id']}")
        else:
            logger.warning(f"Problème lors du traitement de la candidature libre: {response.get('error', 'Erreur inconnue')}")
            
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la candidature libre: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/application-metrics/<application_id>', methods=['GET'])
def get_application_metrics(application_id):
    """
    Endpoint pour récupérer les métriques détaillées d'une candidature.
    Permet de séparer ces données de la réponse principale pour alléger les requêtes.
    """
    try:
        try:
            application = cv_collection.find_one({'_id': ObjectId(application_id)})
            if not application:
                return jsonify({'error': 'Candidature non trouvée'}), 404
        except Exception as e:
            return jsonify({'error': f"ID de candidature invalide: {str(e)}"}), 400
            
        # Récupérer les métriques depuis les données stockées
        metrics = application.get('match_data', {}).get('metrics', {})
        if not metrics:
            # Si aucune métrique n'est trouvée, on peut les recalculer
            entities = application.get('entities', {})
            job_id = application.get('offer_id')
            
            try:
                job_offer = offers_collection.find_one({'_id': ObjectId(job_id)})
                job_keywords = job_offer.get('keywords', [])
                matched_skills = application.get('match_data', {}).get('matched_skills', [])
                
                metrics = calculate_matching_metrics(
                    entities.get('skills', []), 
                    job_keywords,
                    matched_skills
                )
            except Exception as calc_error:
                logger.error(f"Erreur lors du recalcul des métriques: {str(calc_error)}")
                metrics = {
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "accuracy": 0,
                    "loss": 1.0,
                    "error": "Impossible de calculer les métriques"
                }
        
        # Ajouter des métriques supplémentaires si nécessaire
        detailed_metrics = {
            **metrics,
            'application_id': application_id,
            'date_applied': application.get('date_applied', ''),
            'skill_count': len(application.get('entities', {}).get('skills', [])),
            'offer_keyword_count': len(application.get('match_data', {}).get('job_keywords', []))
        }
        
        return jsonify({
            'metrics': detailed_metrics,
            'matching_details': {
                'skill_match_score': application.get('match_data', {}).get('skill_match_score', 0),
                'semantic_similarity': application.get('match_data', {}).get('semantic_similarity', 0),
                'final_score': application.get('match_data', {}).get('final_score', 0)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/generate-test', methods=['POST'])
def generate_job_test():
    """
    Génère un test automatisé basé sur les mots-clés et le titre de l'offre d'emploi.
    Utilise le modèle llama3.2 d'Ollama si disponible, sinon utilise une méthode de secours.
    """
    try:
        logger.info("Requête reçue sur /generate-test")
        
        # Vérifier si les données sont au format JSON
        if not request.is_json:
            return jsonify({'error': "Le type de contenu doit être 'application/json'"}), 415
        
        data = request.json
        keywords = []
        
        # Extraire les mots-clés - peut être dans data['keywords'] ou data['job_offer']['keywords']
        if 'keywords' in data:
            keywords = data['keywords']
        elif 'job_offer' in data and 'keywords' in data['job_offer']:
            keywords = data['job_offer']['keywords']
        else:
            return jsonify({'error': 'Les mots-clés sont requis pour générer un test'}), 400
        
        # S'assurer que keywords est une liste
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        
        if not keywords:
            return jsonify({'error': 'Liste de mots-clés vide'}), 400
        
        # Récupérer le niveau de difficulté
        niveau = data.get('niveau', data.get('job_offer', {}).get('niveau', 'standard')).lower()
        
        # Mapper les niveaux aux difficultés
        difficulty_map = {
            'junior': 'débutant',
            'débutant': 'débutant',
            'standard': 'intermédiaire',
            'confirmé': 'intermédiaire',
            'intermédiaire': 'intermédiaire',
            'senior': 'avancé',
            'expert': 'avancé'
        }
        
        difficulty = difficulty_map.get(niveau, 'intermédiaire')
        
        # Préparer un objet job_offer minimal pour l'API
        job_offer = {
            'keywords': keywords,
            'title': data.get('title', data.get('job_offer', {}).get('title', 'Test de compétences')),
            'niveau': niveau
        }
        
        # Vérifier si Ollama est disponible
        try:
            import ollama
            logger.info("Tentative de génération avec Ollama")
            
            competency_tests = []
            
            # Générer des questions pour chaque mot-clé avec llama3.2
            for keyword in keywords:
                question_prompt = (
                    f"Génère exactement 3 questions à choix multiples sur la compétence '{keyword}', niveau {difficulty}. "
                    f"Format JSON: [{{'question': '...', 'options': ['A: ...', 'B: ...', 'C: ...', 'D: ...'], "
                    f"'correct_answer': 'A: ...', 'explanation': '...'}}]. "
                    f"Réponds uniquement avec du JSON valide."
                )
                
                try:
                    try:
                        # Essayer d'abord avec llama3.2
                        response = ollama.chat(model='llama3.2', messages=[{
                            'role': 'user',
                            'content': question_prompt,
                        }])
                    except:
                        # Sinon essayer avec llama3
                        try:
                            response = ollama.chat(model='llama3', messages=[{
                                'role': 'user',
                                'content': question_prompt,
                            }])
                        except:
                            # Si aucun modèle ne fonctionne, lever une exception
                            raise Exception("Aucun modèle Ollama disponible")
                    
                    response_content = response.get('message', {}).get('content', '')
                    
                    # Extraire le JSON de la réponse
                    import re
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_content)
                    if json_match:
                        json_content = json_match.group(1)
                    else:
                        json_content = response_content
                    
                    # Nettoyer le JSON
                    json_content = json_content.strip()
                    if json_content.startswith('[') and not json_content.endswith(']'):
                        json_content += ']'
                    if not json_content.startswith('['):
                        json_content = '[' + json_content + ']'
                    
                    # Parser le JSON
                    try:
                        questions = json.loads(json_content)
                        
                        # S'assurer que c'est une liste
                        if not isinstance(questions, list):
                            questions = [questions]
                        
                        # Ajouter le mot-clé à chaque question
                        for q in questions:
                            q['keyword'] = keyword
                        
                        competency_tests.extend(questions)
                        
                    except json.JSONDecodeError:
                        # Si le parsing échoue, essayer une approche plus simple
                        logger.warning(f"Échec du parsing JSON pour '{keyword}', tentative simplifiée")
                        
                        # Essayer un prompt plus simple
                        simple_prompt = (
                            f"Crée 2 questions à choix multiples sur {keyword}.\n"
                            f"Pour chaque question, donne:\n"
                            f"1. La question\n"
                            f"2. 4 options (A, B, C, D)\n"
                            f"3. La réponse correcte"
                        )
                        
                        response = ollama.chat(model='llama3', messages=[{
                            'role': 'user',
                            'content': simple_prompt,
                        }])
                        
                        content = response.get('message', {}).get('content', '')
                        
                        # Extraire manuellement les questions
                        lines = content.split('\n')
                        current_question = None
                        options = []
                        correct_answer = None
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            if line.startswith('Question') or re.match(r'^\d+\.', line):
                                # Si on a déjà une question en cours, on la sauvegarde
                                if current_question and options:
                                    competency_tests.append({
                                        'keyword': keyword,
                                        'question': current_question,
                                        'options': options,
                                        'correct_answer': correct_answer or options[0],
                                        'explanation': f"Question sur {keyword}"
                                    })
                                    options = []
                                
                                # Nouvelle question
                                current_question = re.sub(r'^Question\s*\d+:?\s*', '', line)
                                current_question = re.sub(r'^\d+\.\s*', '', current_question)
                                
                            elif line.startswith(('A:', 'B:', 'C:', 'D:')) or line.startswith(('A.', 'B.', 'C.', 'D.')):
                                options.append(line.replace('.', ':') if line[1] == '.' else line)
                                
                            elif 'réponse' in line.lower() or 'correct' in line.lower():
                                # Essayer de trouver la lettre de la réponse
                                for letter in ['A', 'B', 'C', 'D']:
                                    if letter in line:
                                        for opt in options:
                                            if opt.startswith(f"{letter}:"):
                                                correct_answer = opt
                                                break
                        
                        # Ajouter la dernière question
                        if current_question and options:
                            competency_tests.append({
                                'keyword': keyword,
                                'question': current_question,
                                'options': options,
                                'correct_answer': correct_answer or options[0],
                                'explanation': f"Question sur {keyword}"
                            })
                        
                        # Si on n'a pas pu extraire de questions, ajouter une question générique
                        if not competency_tests or len(competency_tests) == 0:
                            raise Exception("Aucune question extraite")
                            
                except Exception as e:
                    logger.error(f"Erreur lors de la génération pour '{keyword}': {str(e)}")
                    # En cas d'erreur, ajouter une question générique
                    competency_tests.append({
                        'keyword': keyword,
                        'question': f"Quelle est la meilleure pratique lors de l'utilisation de {keyword}?",
                        'options': [
                            "A: Suivre la documentation officielle", 
                            "B: Utiliser des bibliothèques tierces", 
                            "C: Se fier uniquement à son expérience", 
                            "D: Ignorer les conventions de nommage"
                        ],
                        'correct_answer': "A: Suivre la documentation officielle",
                        'explanation': f"Il est toujours recommandé de suivre la documentation officielle pour {keyword}."
                    })
            
            # Créer le test final
            test_data = {
                "title": f"Test technique pour {job_offer.get('title', 'poste non spécifié')}",
                "niveau": niveau,
                "qcm": competency_tests,
                "keywords": keywords,
                "generated_by": "llama3.2"
            }
            
            return jsonify({
                'test': test_data,
                'job_offer': {
                    'title': job_offer.get('title'),
                    'keywords': job_offer.get('keywords'),
                    'niveau': job_offer.get('niveau')
                }
            }), 200
            
        except ImportError:
            # Ollama n'est pas disponible, utiliser le système de secours
            logger.warning("Ollama n'est pas disponible, utilisation du mode de secours")
        except Exception as ollama_error:
            # Une erreur s'est produite avec Ollama, utiliser le système de secours
            logger.error(f"Erreur avec Ollama: {str(ollama_error)}, utilisation du mode de secours")
            
        # ====== MODE DE SECOURS (sans Ollama) ======
        logger.info("Génération de questions avec le système de secours")
        
        # Générer des questions pour tous les mots-clés
        all_questions = []
        
        # Ajouter au moins une question par mot-clé
        for keyword in keywords:
            # Question générique 1
            all_questions.append({
                'keyword': keyword,
                'question': f"Quelle est l'utilisation principale de {keyword}?",
                'options': [
                    "A: Développement frontend", 
                    "B: Développement backend", 
                    "C: Analyses de données", 
                    "D: Toutes ces réponses selon le contexte"
                ],
                'correct_answer': "D: Toutes ces réponses selon le contexte",
                'explanation': f"{keyword} peut être utilisé dans différents contextes selon les besoins spécifiques."
            })
            
            # Question générique 2
            all_questions.append({
                'keyword': keyword,
                'question': f"Quel est l'avantage principal de {keyword} par rapport à ses alternatives?",
                'options': [
                    "A: Performance", 
                    "B: Facilité d'utilisation", 
                    "C: Écosystème riche", 
                    "D: Support communautaire"
                ],
                'correct_answer': "C: Écosystème riche",
                'explanation': f"{keyword} est souvent reconnu pour son écosystème riche et ses nombreuses possibilités."
            })
        
        # Créer le test final avec le système de secours
        test_data = {
            "title": f"Test technique pour {job_offer.get('title', 'poste non spécifié')}",
            "niveau": niveau,
            "qcm": all_questions,
            "keywords": keywords,
            "generated_by": "système de secours (sans Ollama)"
        }
        
        return jsonify({
            'test': test_data,
            'job_offer': {
                'title': job_offer.get('title'),
                'keywords': job_offer.get('keywords'),
                'niveau': job_offer.get('niveau')
            },
            'note': "Pour générer des questions plus personnalisées, installez Ollama avec: curl -fsSL https://ollama.ai/install.sh | sh"
        }), 200
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération du test: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def generate_with_ollama(keywords, difficulty, job_offer):
    """
    Génère des questions de test en utilisant le modèle Ollama llama3.2
    """
    import ollama
    competency_tests = []
    
    # Générer des questions pour chaque mot-clé
    for keyword in keywords:
        question_prompt = (
            f"Génère exactement 3 questions à choix multiples sur la compétence '{keyword}', niveau {difficulty}. "
            f"Format JSON: [{{'question': '...', 'options': ['A: ...', 'B: ...', 'C: ...', 'D: ...'], "
            f"'correct_answer': 'A: ...', 'explanation': '...'}}]. "
            f"Réponds uniquement avec du JSON valide."
        )
        
        try:
            # Essayer d'abord llama3.2, puis llama3 si échoue
            try:
                response = ollama.chat(model='llama3.2', messages=[{
                    'role': 'user',
                    'content': question_prompt,
                }])
            except:
                response = ollama.chat(model='llama3', messages=[{
                    'role': 'user',
                    'content': question_prompt,
                }])
            
            response_content = response.get('message', {}).get('content', '')
            
            # Extraire le JSON de la réponse
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_content)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_content = response_content
            
            # Nettoyer le JSON
            json_content = json_content.strip()
            if json_content.startswith('[') and not json_content.endswith(']'):
                json_content += ']'
            if not json_content.startswith('['):
                json_content = '[' + json_content + ']'
            
            # Parser le JSON
            try:
                questions = json.loads(json_content)
                
                # S'assurer que c'est une liste
                if not isinstance(questions, list):
                    questions = [questions]
                
                # Ajouter le mot-clé à chaque question
                for q in questions:
                    q['keyword'] = keyword
                
                competency_tests.extend(questions)
                
            except json.JSONDecodeError:
                # Si le parsing échoue, essayer une approche plus simple
                logger.warning(f"Échec du parsing JSON pour '{keyword}', tentative simplifiée")
                
                # Essayer un prompt plus simple
                simple_prompt = (
                    f"Crée 2 questions à choix multiples sur {keyword}.\n"
                    f"Pour chaque question, donne:\n"
                    f"1. La question\n"
                    f"2. 4 options (A, B, C, D)\n"
                    f"3. La réponse correcte"
                )
                
                try:
                    response = ollama.chat(model='llama3', messages=[{
                        'role': 'user',
                        'content': simple_prompt,
                    }])
                    
                    content = response.get('message', {}).get('content', '')
                    
                    # Extraire manuellement les questions
                    lines = content.split('\n')
                    current_question = None
                    options = []
                    correct_answer = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        if line.startswith('Question') or re.match(r'^\d+\.', line):
                            # Si on a déjà une question en cours, on la sauvegarde
                            if current_question and options:
                                competency_tests.append({
                                    'keyword': keyword,
                                    'question': current_question,
                                    'options': options,
                                    'correct_answer': correct_answer or options[0],
                                    'explanation': f"Question sur {keyword}"
                                })
                                options = []
                            
                            # Nouvelle question
                            current_question = re.sub(r'^Question\s*\d+:?\s*', '', line)
                            current_question = re.sub(r'^\d+\.\s*', '', current_question)
                            
                        elif line.startswith(('A:', 'B:', 'C:', 'D:')) or line.startswith(('A.', 'B.', 'C.', 'D.')):
                            options.append(line.replace('.', ':') if line[1] == '.' else line)
                            
                        elif 'réponse' in line.lower() or 'correct' in line.lower():
                            # Essayer de trouver la lettre de la réponse
                            for letter in ['A', 'B', 'C', 'D']:
                                if letter in line:
                                    for opt in options:
                                        if opt.startswith(f"{letter}:"):
                                            correct_answer = opt
                                            break
                    
                    # Ajouter la dernière question
                    if current_question and options:
                        competency_tests.append({
                            'keyword': keyword,
                            'question': current_question,
                            'options': options,
                            'correct_answer': correct_answer or options[0],
                            'explanation': f"Question sur {keyword}"
                        })
                        
                except Exception as parsing_error:
                    logger.error(f"Erreur lors du parsing des questions: {str(parsing_error)}")
                    # Ajouter une question de secours
                    competency_tests.append(generate_basic_question(keyword))
        except Exception as e:
            logger.error(f"Erreur lors de la génération pour '{keyword}': {str(e)}")
            # En cas d'erreur, ajouter une question générique
            competency_tests.append(generate_basic_question(keyword))
    
    # Créer le test final
    test_data = {
        "title": f"Test technique pour {job_offer.get('title', 'poste non spécifié')}",
        "niveau": job_offer.get('niveau', 'intermédiaire'),
        "qcm": competency_tests,
        "keywords": keywords,
        "generated_by": "llama3.2"
    }
    
    return jsonify({
        'test': test_data,
        'job_offer': {
            'title': job_offer.get('title'),
            'keywords': job_offer.get('keywords'),
            'niveau': job_offer.get('niveau')
        }
    }), 200

def generate_fallback_questions(keywords, niveau, job_offer):
    """
    Génère des questions de test basiques sans dépendre d'Ollama
    """
    logger.info("Génération de questions de secours sans Ollama")
    
    # Questions de base par compétence
    all_questions = []
    
    # Générer au moins une question par mot-clé
    for keyword in keywords:
        # Ajouter des questions spécifiques au mot-clé
        keyword_questions = generate_questions_for_keyword(keyword, niveau)
        all_questions.extend(keyword_questions)
    
    # Créer le test final
    test_data = {
        "title": f"Test technique pour {job_offer.get('title', 'poste non spécifié')}",
        "niveau": niveau,
        "qcm": all_questions,
        "keywords": keywords,
        "generated_by": "système de secours (sans Ollama)"
    }
    
    return jsonify({
        'test': test_data,
        'job_offer': {
            'title': job_offer.get('title'),
            'keywords': job_offer.get('keywords'),
            'niveau': job_offer.get('niveau')
        },
        'note': "Pour générer des questions plus personnalisées, installez Ollama avec: curl -fsSL https://ollama.ai/install.sh | sh"
    }), 200

def generate_questions_for_keyword(keyword, niveau):
    """
    Génère des questions spécifiques pour un mot-clé donné
    """
    # Questions génériques adaptables
    questions = []
    
    # Questions universelles qui peuvent être adaptées à n'importe quel mot-clé
    base_templates = [
        {
            "question_template": "Quelle est l'utilisation principale de {keyword}?",
            "options": [
                "A: Développement frontend", 
                "B: Développement backend", 
                "C: Analyses de données", 
                "D: Toutes ces réponses selon le contexte"
            ],
            "correct_answer": "D: Toutes ces réponses selon le contexte",
            "explanation_template": "{keyword} peut être utilisé dans différents contextes selon les besoins spécifiques."
        },
        {
            "question_template": "Quel est l'avantage principal de {keyword} par rapport à ses alternatives?",
            "options": [
                "A: Performance", 
                "B: Facilité d'utilisation", 
                "C: Écosystème riche", 
                "D: Support communautaire"
            ],
            "correct_answer": "C: Écosystème riche",
            "explanation_template": "{keyword} est souvent reconnu pour son écosystème riche et ses nombreuses possibilités."
        },
        {
            "question_template": "Dans quel contexte {keyword} est-il particulièrement recommandé?",
            "options": [
                "A: Petits projets", 
                "B: Projets d'entreprise", 
                "C: Applications mobiles", 
                "D: Applications web"
            ],
            "correct_answer": "D: Applications web",
            "explanation_template": "{keyword} est fréquemment utilisé dans le développement d'applications web modernes."
        }
    ]
    
    # Questions spécifiques à certains domaines
    domain_specific = {
        # Programmation
        "python": [
            {
                "question": "Quel type de langage est Python?",
                "options": [
                    "A: Langage compilé", 
                    "B: Langage interprété", 
                    "C: Langage assembleur", 
                    "D: Langage binaire"
                ],
                "correct_answer": "B: Langage interprété",
                "explanation": "Python est un langage interprété, ce qui signifie que le code est exécuté directement sans compilation préalable."
            }
        ],
        "java": [
            {
                "question": "Quelle fonctionnalité est au cœur de Java?",
                "options": [
                    "A: Machine virtuelle (JVM)", 
                    "B: Compilation en code natif", 
                    "C: Typage dynamique", 
                    "D: Gestion manuelle de la mémoire"
                ],
                "correct_answer": "A: Machine virtuelle (JVM)",
                "explanation": "La machine virtuelle Java (JVM) permet à Java d'être indépendant de la plateforme."
            }
        ],
        "javascript": [
            {
                "question": "Quelle est la particularité principale de JavaScript?",
                "options": [
                    "A: Il peut s'exécuter uniquement côté serveur", 
                    "B: Il s'exécute dans le navigateur", 
                    "C: Il nécessite une compilation", 
                    "D: Il est fortement typé"
                ],
                "correct_answer": "B: Il s'exécute dans le navigateur",
                "explanation": "JavaScript a été conçu pour s'exécuter côté client dans les navigateurs web."
            }
        ],
        
        # Web
        "html": [
            {
                "question": "Que signifie HTML?",
                "options": [
                    "A: HyperText Markup Language", 
                    "B: High-level Text Management Language", 
                    "C: Home Tool Markup Language", 
                    "D: Hybrid Text Manipulation Layer"
                ],
                "correct_answer": "A: HyperText Markup Language",
                "explanation": "HTML (HyperText Markup Language) est le langage standard pour créer des pages web."
            }
        ],
        "css": [
            {
                "question": "Quel est le rôle principal de CSS?",
                "options": [
                    "A: Gérer la logique d'une page web", 
                    "B: Définir la structure d'une page web", 
                    "C: Gérer le style et l'apparence", 
                    "D: Gérer les interactions utilisateur"
                ],
                "correct_answer": "C: Gérer le style et l'apparence",
                "explanation": "CSS (Cascading Style Sheets) est utilisé pour contrôler la présentation et le style des éléments HTML."
            }
        ],
        
        # Frameworks & Bibliothèques
        "react": [
            {
                "question": "Quel concept est fondamental en React?",
                "options": [
                    "A: Les contrôleurs", 
                    "B: Les composants", 
                    "C: Les services", 
                    "D: Les directives"
                ],
                "correct_answer": "B: Les composants",
                "explanation": "React est basé sur l'architecture à composants, où l'interface est divisée en éléments réutilisables."
            }
        ],
        "angular": [
            {
                "question": "Qu'est-ce qui distingue Angular?",
                "options": [
                    "A: C'est une bibliothèque légère", 
                    "B: C'est un framework complet", 
                    "C: Il utilise uniquement JavaScript", 
                    "D: Il ne gère que le CSS"
                ],
                "correct_answer": "B: C'est un framework complet",
                "explanation": "Angular est un framework complet qui fournit tout ce dont vous avez besoin pour construire des applications SPA."
            }
        ],
        
        # Backend & Serveurs
        "node.js": [
            {
                "question": "Quelle est la particularité de Node.js?",
                "options": [
                    "A: Il exécute JavaScript côté serveur", 
                    "B: Il est écrit en C++", 
                    "C: Il ne peut gérer qu'une requête à la fois", 
                    "D: Il est réservé au développement mobile"
                ],
                "correct_answer": "A: Il exécute JavaScript côté serveur",
                "explanation": "Node.js permet d'exécuter du code JavaScript côté serveur, en dehors d'un navigateur."
            }
        ],
        
        # Bases de données
        "sql": [
            {
                "question": "Que signifie SQL?",
                "options": [
                    "A: Simple Query Language", 
                    "B: Structured Query Language", 
                    "C: Sequential Query Language", 
                    "D: Secure Query Logic"
                ],
                "correct_answer": "B: Structured Query Language",
                "explanation": "SQL (Structured Query Language) est un langage de programmation conçu pour gérer les données dans les systèmes de gestion de bases de données relationnelles."
            }
        ],
        "mongodb": [
            {
                "question": "Quel type de base de données est MongoDB?",
                "options": [
                    "A: Relationnelle", 
                    "B: NoSQL orientée documents", 
                    "C: Graphe", 
                    "D: Hiérarchique"
                ],
                "correct_answer": "B: NoSQL orientée documents",
                "explanation": "MongoDB est une base de données NoSQL orientée documents qui stocke les données au format BSON (JSON binaire)."
            }
        ],
        
        # DevOps
        "docker": [
            {
                "question": "Quelle technologie est au cœur de Docker?",
                "options": [
                    "A: Virtualisation complète", 
                    "B: Conteneurisation", 
                    "C: Hyperviseurs de type 1", 
                    "D: Machines virtuelles Java"
                ],
                "correct_answer": "B: Conteneurisation",
                "explanation": "Docker utilise la conteneurisation pour empaqueter des applications avec leurs dépendances dans des unités standardisées."
            }
        ],
        "kubernetes": [
            {
                "question": "Quel est le rôle principal de Kubernetes?",
                "options": [
                    "A: Développement d'applications", 
                    "B: Orchestration de conteneurs", 
                    "C: Virtualisation de serveurs", 
                    "D: Gestion de bases de données"
                ],
                "correct_answer": "B: Orchestration de conteneurs",
                "explanation": "Kubernetes est une plateforme d'orchestration de conteneurs qui automatise le déploiement, la mise à l'échelle et la gestion des applications conteneurisées."
            }
        ]
    }
    
    # Chercher des questions spécifiques au mot-clé
    if keyword.lower() in domain_specific:
        specific_questions = domain_specific[keyword.lower()]
        questions.extend([{**q, 'keyword': keyword} for q in specific_questions])
    
    # Si pas assez de questions spécifiques, ajouter des questions génériques
    if len(questions) < 2:
        # Chercher des correspondances partielles
        for k, qs in domain_specific.items():
            if k in keyword.lower() or keyword.lower() in k:
                for q in qs:
                    adapted_q = {
                        **q,
                        'keyword': keyword,
                        'question': q['question'].replace(k, keyword),
                        'explanation': q['explanation'].replace(k, keyword)
                    }
                    questions.append(adapted_q)
                    if len(questions) >= 2:
                        break
            if len(questions) >= 2:
                break
    
    # Si toujours pas assez de questions, utiliser les templates génériques
    if len(questions) < 2:
        for template in base_templates:
            questions.append({
                'keyword': keyword,
                'question': template['question_template'].format(keyword=keyword),
                'options': template['options'],
                'correct_answer': template['correct_answer'],
                'explanation': template['explanation_template'].format(keyword=keyword)
            })
            if len(questions) >= 3:
                break
    
    return questions

def generate_basic_question(keyword):
    """
    Génère une question basique pour un mot-clé donné
    """
    return {
        'keyword': keyword,
        'question': f"Quelle est la meilleure pratique lors de l'utilisation de {keyword}?",
        'options': [
            "A: Suivre la documentation officielle", 
            "B: Utiliser des bibliothèques tierces", 
            "C: Se fier uniquement à son expérience", 
            "D: Ignorer les conventions de nommage"
        ],
        'correct_answer': "A: Suivre la documentation officielle",
        'explanation': f"Il est toujours recommandé de suivre la documentation officielle pour {keyword}."
    }

# Amélioration de la détection du nom pour éviter de confondre avec des sections de CV
def extract_name_improved(cv_text):
    """
    Extrait le nom du candidat en évitant de confondre avec des titres de section.
    
    Args:
        cv_text (str): Le texte du CV
        
    Returns:
        str: Le nom détecté ou None
    """
    # Liste de titres de section courants à éviter
    section_titles = [
        "education", "formation", "expérience", "experience", "compétences", 
        "skills", "profil", "profile", "résumé", "summary", "contact", "coordonnées",
        "projets", "projects", "languages", "langues", "certifications", "références"
    ]
    
    # Essayer de trouver le nom au début du CV
    lines = cv_text.split('\n')
    potential_names = []
    
    # Analyser les 10 premières lignes non vides
    non_empty_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        non_empty_count += 1
        if non_empty_count > 10:
            break
            
        # Ignorer les lignes trop courtes ou trop longues
        if len(line) < 3 or len(line) > 40:
            continue
            
        # Ignorer les lignes qui sont des titres de section
        if line.lower() in section_titles or any(title in line.lower() for title in section_titles):
            continue
            
        # Ignorer les lignes qui contiennent des caractères spéciaux courants dans les emails/téléphones/adresses
        if any(char in line for char in ['@', ':', '/', '\\', 'http']):
            continue
            
        # Ignorer les lignes qui commencent par des caractères non alphabétiques
        if line and not line[0].isalpha():
            continue
            
        # Les noms ont tendance à avoir des majuscules
        words = line.split()
        if any(word and word[0].isupper() for word in words):
            potential_names.append(line)
    
    # Prioriser les noms qui ont l'air d'être des noms propres (commençant par majuscule)
    for name in potential_names:
        words = name.split()
        # Vérifier si le nom ressemble à un nom propre (2-3 mots commençant par une majuscule)
        if 1 <= len(words) <= 3 and all(word and word[0].isupper() for word in words):
            return name
    
    # Si aucun nom propre évident n'est trouvé, prendre le premier potentiel
    if potential_names:
        return potential_names[0]
        
    return None

def calculate_semantic_similarity_enhanced(text1, text2):
    """
    Calcule une similarité sémantique améliorée entre deux textes.
    Utilise une approche combinée avec embedding et extraction de mots-clés.
    
    Args:
        text1 (str): Premier texte à comparer
        text2 (str): Second texte à comparer
        
    Returns:
        float: Score de similarité entre 0 et 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Tronquer les textes pour éviter les problèmes de mémoire
    max_length = 5000
    text1 = text1[:max_length] if len(text1) > max_length else text1
    text2 = text2[:max_length] if len(text2) > max_length else text2
    
    # 1. Similarité par embedding
    try:
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)
        cosine_sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
    except Exception as e:
        logger.warning(f"Erreur lors du calcul de la similarité par embedding: {str(e)}")
        cosine_sim = 0.0
    
    # 2. Similarité basée sur les mots-clés/termes importants
    try:
        # Extraire les termes importants des deux textes
        keywords1 = set(optimize_skill_extraction(text1))
        keywords2 = set(optimize_skill_extraction(text2))
        
        # Calculer la similarité de Jaccard
        if keywords1 and keywords2:
            intersection = keywords1.intersection(keywords2)
            union = keywords1.union(keywords2)
            jaccard_sim = len(intersection) / len(union) if union else 0
        else:
            jaccard_sim = 0.0
    except Exception as e:
        logger.warning(f"Erreur lors du calcul de la similarité par mots-clés: {str(e)}")
        jaccard_sim = 0.0
    
    # 3. Similarité basée sur le nombre de mots-clés communs (métrique simple mais efficace)
    try:
        common_keywords = keywords1.intersection(keywords2)
        keyword_count_sim = len(common_keywords) / max(len(keywords1), len(keywords2)) if max(len(keywords1), len(keywords2)) > 0 else 0
    except Exception as e:
        logger.warning(f"Erreur lors du calcul de la similarité par mots communs: {str(e)}")
        keyword_count_sim = 0.0
    
    # 4. Combinaison pondérée des similarités
    # Donner plus de poids à l'embedding qui est généralement plus sophistiqué
    final_similarity = (cosine_sim * 0.6) + (jaccard_sim * 0.2) + (keyword_count_sim * 0.2)
    
    return max(0.0, min(1.0, final_similarity))  # Garantir que la valeur est entre 0 et 1

# Initialize Flask routes
def init_routes(app):
    # Register the Blueprint with the app
    
    app.register_blueprint(bp, url_prefix='/api')

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Module ollama non installé, utilisation du système de secours")