"""
Module pour la gestion des candidatures libres (sans identifiant d'offre spécifique).
Le système trouve automatiquement l'offre d'emploi la plus adaptée au profil du candidat.
"""
import logging
import numpy as np
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenApplicationProcessor:
    """
    Classe pour traiter les candidatures libres et trouver les meilleures offres correspondantes
    """
    
    def __init__(self, cv_collection, offers_collection, sentence_model):
        """
        Initialise le processeur de candidatures libres
        
        Args:
            cv_collection: Collection MongoDB pour les CVs
            offers_collection: Collection MongoDB pour les offres
            sentence_model: Modèle SentenceTransformer pour l'encodage du texte
        """
        self.cv_collection = cv_collection
        self.offers_collection = offers_collection
        self.sentence_model = sentence_model
    
    def get_embedding(self, text):
        """
        Génère un embedding vectoriel à partir d'un texte
        
        Args:
            text (str): Texte à encoder
            
        Returns:
            numpy.ndarray: Vecteur d'embedding
        """
        if not text or text.strip() == "":
            return np.zeros(384)  # Taille par défaut de l'embedding
        
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length]
            
        with torch.no_grad():
            embedding = self.sentence_model.encode(text)
            
        return embedding
    
    def calculate_semantic_similarity(self, embedding1, embedding2):
        """
        Calcule la similarité cosinus entre deux embeddings
        
        Args:
            embedding1 (numpy.ndarray): Premier embedding
            embedding2 (numpy.ndarray): Second embedding
            
        Returns:
            float: Score de similarité entre 0 et 1
        """
        embedding1_reshaped = embedding1.reshape(1, -1)
        embedding2_reshaped = embedding2.reshape(1, -1)
        similarity = cosine_similarity(embedding1_reshaped, embedding2_reshaped)[0][0]
        return similarity
    
    def find_best_matching_offers(self, cv_text, cv_skills, top_n=3):
        """
        Trouve les meilleures offres d'emploi correspondant au CV
        
        Args:
            cv_text (str): Texte complet du CV
            cv_skills (list): Compétences extraites du CV
            top_n (int): Nombre d'offres à retourner
            
        Returns:
            list: Liste des meilleures offres avec leurs scores
        """
        cv_embedding = self.get_embedding(cv_text)
        matches = []
        
        # Récupérer toutes les offres de la base de données
        all_offers = list(self.offers_collection.find())
        
        if not all_offers:
            return []
        
        for offer in all_offers:
            # Récupérer la description et les mots-clés de l'offre
            job_description = offer.get('description', '')
            job_keywords = offer.get('keywords', [])
            
            # Calculer le score de similarité sémantique
            job_embedding = self.get_embedding(job_description)
            semantic_similarity = self.calculate_semantic_similarity(cv_embedding, job_embedding)
            
            # Calculer le score de correspondance des compétences
            skill_match_score, matched_skills = self.calculate_skill_match(cv_skills, job_keywords)
            
            # Pondération : 70% compétences, 30% similarité sémantique
            combined_score = (skill_match_score * 0.7) + (semantic_similarity * 100 * 0.3)
            
            # Ajouter les informations de correspondance
            matches.append({
                'offer': offer,
                'score': round(combined_score, 1),
                'semantic_similarity': round(semantic_similarity * 100, 1),
                'skill_match_score': skill_match_score,
                'matched_skills': matched_skills
            })
        
        # Trier par score décroissant et prendre les top_n
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_n]
    
    def calculate_skill_match(self, cv_skills, job_keywords):
        """
        Calcule le score de correspondance entre les compétences du CV et les mots-clés de l'offre
        
        Args:
            cv_skills (list): Compétences extraites du CV
            job_keywords (list): Mots-clés de l'offre d'emploi
            
        Returns:
            tuple: (score, matched_skills)
        """
        # Si l'une des listes est vide, retourner un score minimum
        if not cv_skills or not job_keywords:
            return 10, []
        
        # Convertir en minuscules pour une comparaison insensible à la casse
        cv_skills_lower = [skill.lower() for skill in cv_skills]
        job_keywords_lower = [kw.lower() for kw in job_keywords]
        
        # Trouver les correspondances exactes
        exact_matches = []
        for skill in cv_skills:
            skill_lower = skill.lower()
            # Vérifier la correspondance exacte
            if skill_lower in job_keywords_lower:
                exact_matches.append(skill)
                continue
                
            # Vérifier la correspondance par sous-chaîne
            for keyword in job_keywords_lower:
                if skill_lower in keyword or keyword in skill_lower:
                    exact_matches.append(skill)
                    break
        
        # Calculer le pourcentage de correspondance directe
        exact_match_pct = (len(exact_matches) / len(job_keywords)) * 100 if job_keywords else 0
        
        # Trouver les correspondances sémantiques
        semantic_matches = []
        semantic_scores = []
        
        for skill in cv_skills:
            if skill in exact_matches:
                continue  # Ignorer si déjà une correspondance exacte
                
            skill_embedding = self.get_embedding(skill.lower())
            best_match_score = 0
            best_match_keyword = None
            
            for keyword in job_keywords:
                keyword_embedding = self.get_embedding(keyword.lower())
                similarity = self.calculate_semantic_similarity(skill_embedding, keyword_embedding)
                
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_keyword = keyword
            
            # Ne comptabiliser que les correspondances sémantiques fortes
            if best_match_score > 0.6:  # Seuil pour une correspondance sémantique
                semantic_matches.append(skill)
                semantic_scores.append(best_match_score)
        
        # Calculer le pourcentage de correspondance sémantique
        semantic_match_pct = 0
        if job_keywords and semantic_scores:
            semantic_match_pct = (sum(semantic_scores) / len(job_keywords)) * 100
        
        # Combiner les correspondances exactes et sémantiques
        all_matches = exact_matches + semantic_matches
        
        # Calculer le score global
        # Donner un poids plus élevé aux correspondances exactes (70%) et plus faible aux correspondances sémantiques (30%)
        match_score = (exact_match_pct * 0.7) + (semantic_match_pct * 0.3)
        
        # Appliquer un bonus pour avoir plusieurs correspondances
        match_bonus = min(40, len(all_matches) * 8)  # 8% par correspondance jusqu'à 40%
        
        # Calculer le score final
        final_score = match_score + match_bonus
        
        # Assurer que le score est entre 10 et 100
        final_score = max(10, min(100, final_score))
        
        return round(final_score, 1), all_matches
    
    def process_open_application(self, cv_text, entities, sanitized_cv_text, personal_info, optimized_resume, metrics_calculator):
        """
        Traite une candidature libre pour trouver la meilleure offre correspondante
        
        Args:
            cv_text (str): Texte complet du CV
            entities (dict): Entités extraites du CV
            sanitized_cv_text (str): Texte du CV nettoyé
            personal_info (dict): Informations personnelles extraites
            optimized_resume (str): Résumé optimisé
            metrics_calculator (function): Fonction pour calculer les métriques
            
        Returns:
            dict: Réponse contenant les détails de l'application et l'offre correspondante
        """
        # Récupérer les compétences extraites
        extracted_skills = entities.get('skills', [])
        
        # Trouver les meilleures offres correspondantes
        best_matches = self.find_best_matching_offers(cv_text, extracted_skills)
        
        if not best_matches:
            return {"error": "Aucune offre d'emploi correspondante trouvée"}, 404
        
        # Utiliser la meilleure offre
        best_match = best_matches[0]
        best_offer = best_match['offer']
        match_score = best_match['score']
        matched_skills = best_match['matched_skills']
        semantic_similarity = best_match['semantic_similarity']
        
        # Convertir l'ObjectId en chaîne
        best_offer['_id'] = str(best_offer['_id'])
        
        # Calculer les métriques détaillées
        job_keywords = best_offer.get('keywords', [])
        metrics = metrics_calculator(extracted_skills, job_keywords, matched_skills)
        
        # Enregistrer la candidature dans la base de données
        application_id = self.cv_collection.insert_one({
            'offer_id': best_offer['_id'],
            'cv_text': sanitized_cv_text,
            'entities': entities,
            'personal_info': personal_info,
            'optimized_resume': optimized_resume,
            'match_data': {
                'skill_match_score': match_score,
                'semantic_similarity': semantic_similarity,
                'final_score': match_score,
                'matched_skills': matched_skills,
                'job_keywords': job_keywords,
                'metrics': metrics,
                'auto_matched': True
            },
            'date_applied': datetime.now()
        }).inserted_id
        
        application_id = str(application_id)
        
        # Préparer les autres offres possibles
        alternative_offers = []
        if len(best_matches) > 1:
            for match in best_matches[1:]:
                offer = match['offer']
                alternative_offers.append({
                    'id': str(offer['_id']),
                    'title': offer.get('title', 'Non spécifié'),
                    'company': offer.get('company', 'Non spécifié'),
                    'score': match['score']
                })
        
        # Préparer la réponse
        response = {
            'application_id': application_id,
            'candidate': {
                'personal_info': personal_info,
                'optimized_resume': optimized_resume,
                'skills': entities.get('skills', [])
            },
            'matched_job_offer': {
                'id': best_offer['_id'],
                'title': best_offer.get('title', 'Non spécifié'),
                'company': best_offer.get('company', 'Non spécifié'),
                'keywords': job_keywords
            },
            'matching': {
                'score': match_score,
                'semantic_similarity': semantic_similarity,
                'matched_skills': matched_skills,
                'metrics': metrics
            },
            'alternative_offers': alternative_offers
        }
        
        return response, 200 