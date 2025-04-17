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
        
        # Supprimer les données sensibles du texte du CV
        sanitized_cv_text = remove_sensitive_data(cv_text)
        
        # Extraire les informations personnelles
        personal_info = {
            'name': entities.get('name', 'Nom non détecté'),
            'email': entities.get('email', 'Email non détecté'),
            'telephone': entities.get('telephone', 'Téléphone non détecté')
        }
        
        # Générer un résumé optimisé
        experience = entities.get('experience', 'Non spécifié')
        optimized_resume = generate_candidate_summary(cv_text, experience)
        
        # Calculer la correspondance des compétences et la similarité sémantique
        skill_match_score, matched_skills = calculate_skill_match(entities.get('skills', []), job_offer.get('keywords', []))
        semantic_similarity = calculate_semantic_similarity(
            get_embedding(cv_text),
            get_embedding(job_offer.get('description', ''))
        ) * 100
        
        # Calculer le score final
        final_score = (skill_match_score * 0.6) + (semantic_similarity * 0.4)
        final_score = max(10, min(final_score, 100))  # Limiter entre 10 et 100
        
        application_id = cv_collection.insert_one({
            'offer_id': offer_id,
            'cv_text': sanitized_cv_text,
            'entities': entities,
            'personal_info': personal_info,
            'optimized_resume': optimized_resume,
            'match_data': {
                'skill_match_score': skill_match_score,
                'semantic_similarity': semantic_similarity,
                'final_score': final_score
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
                'keywords': job_offer.get('keywords', [])
            },
            'matching': {
                'score': final_score,
                'skill_match_score': skill_match_score,
                'semantic_similarity': semantic_similarity,
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

# Initialize Flask routes
def init_routes(app):
    # Register the Blueprint with the app
    app.register_blueprint(bp, url_prefix='/api')