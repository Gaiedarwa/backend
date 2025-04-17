import os
import logging
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Obtenir le modèle à utiliser depuis les variables d'environnement ou utiliser un modèle par défaut
MODEL_NAME = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')

# Cache pour les embeddings, pour éviter de recalculer les embeddings fréquemment utilisés
embedding_cache = {}
model = None

def get_model():
    """Récupérer l'instance du modèle SentenceTransformer (avec lazy loading)"""
    global model
    if model is None:
        try:
            logger.info(f"Chargement du modèle SentenceTransformer: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            logger.info("Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
    return model

def get_embedding(text, use_cache=True):
    """
    Obtenir l'embedding d'un texte
    
    Args:
        text (str): Texte à encoder
        use_cache (bool): Utiliser le cache pour éviter les calculs redondants
        
    Returns:
        numpy.ndarray: Vecteur d'embedding
    """
    if not text:
        # Retourner un vecteur de zéros de la même dimension que le modèle
        dim = 384  # Dimension par défaut pour all-MiniLM-L6-v2
        return np.zeros(dim)
    
    # Normaliser le texte
    text = text.strip().lower()
    
    # Vérifier le cache
    if use_cache and text in embedding_cache:
        return embedding_cache[text]
    
    try:
        # Obtenir le modèle et calculer l'embedding
        model = get_model()
        embedding = model.encode(text)
        
        # Mettre en cache si nécessaire
        if use_cache:
            embedding_cache[text] = embedding
        
        return embedding
    except Exception as e:
        logger.error(f"Erreur lors du calcul de l'embedding: {str(e)}")
        # Méthode de secours en cas d'erreur
        dim = 384  # Dimension par défaut pour all-MiniLM-L6-v2
        return np.zeros(dim)

def batch_get_embeddings(texts, use_cache=True):
    """
    Obtenir les embeddings pour une liste de textes (plus efficace que les appels individuels)
    
    Args:
        texts (list): Liste de textes à encoder
        use_cache (bool): Utiliser le cache
        
    Returns:
        list: Liste des vecteurs d'embedding
    """
    # Filtrer les textes vides et normaliser
    texts = [text.strip().lower() for text in texts if text]
    
    if not texts:
        return []
    
    # Vérifier quels textes sont dans le cache
    cache_hits = []
    texts_to_encode = []
    indices = []
    
    for i, text in enumerate(texts):
        if use_cache and text in embedding_cache:
            cache_hits.append((i, embedding_cache[text]))
        else:
            texts_to_encode.append(text)
            indices.append(i)
    
    # Calculer les embeddings pour les textes non mis en cache
    embeddings = []
    if texts_to_encode:
        try:
            model = get_model()
            batch_embeddings = model.encode(texts_to_encode)
            
            # Mettre en cache les nouveaux embeddings
            if use_cache:
                for text, emb in zip(texts_to_encode, batch_embeddings):
                    embedding_cache[text] = emb
                    
            # Combiner avec les embeddings du cache
            embeddings = [None] * len(texts)
            for i, emb in cache_hits:
                embeddings[i] = emb
            for idx, emb in zip(indices, batch_embeddings):
                embeddings[idx] = emb
                
        except Exception as e:
            logger.error(f"Erreur lors du calcul des embeddings par lots: {str(e)}")
            # Méthode de secours en cas d'erreur
            dim = 384  # Dimension par défaut
            embeddings = [np.zeros(dim) for _ in range(len(texts))]
    else:
        # Tous les embeddings étaient dans le cache
        embeddings = [None] * len(texts)
        for i, emb in cache_hits:
            embeddings[i] = emb
    
    return embeddings

def calculate_semantic_similarity(text1, text2):
    """
    Calculer la similarité sémantique entre deux textes
    
    Args:
        text1 (str): Premier texte
        text2 (str): Deuxième texte
        
    Returns:
        float: Score de similarité (0-100)
    """
    if not text1 or not text2:
        return 0
    
    # Obtenir les embeddings
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    # Calculer la similarité cosinus
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    # Convertir en pourcentage (0-100)
    return float(similarity * 100)

def calculate_skills_similarity_semantic(skills1, skills2):
    """
    Calculer la similarité sémantique entre deux ensembles de compétences
    
    Args:
        skills1 (list): Premier ensemble de compétences
        skills2 (list): Deuxième ensemble de compétences
        
    Returns:
        float: Score de similarité (0-100)
    """
    if not skills1 or not skills2:
        return 0
    
    # Combiner les compétences en textes
    text1 = " ".join(skills1)
    text2 = " ".join(skills2)
    
    # Calculer la similarité sémantique
    return calculate_semantic_similarity(text1, text2)

def calculate_skills_similarity(skills1, skills2):
    """
    Calculer la similarité entre deux ensembles de compétences
    en combinant approche lexicale et sémantique
    
    Args:
        skills1 (list): Premier ensemble de compétences
        skills2 (list): Deuxième ensemble de compétences
        
    Returns:
        float: Score de similarité (0-100)
    """
    if not skills1 or not skills2:
        return 0
    
    # Normaliser les compétences
    skills1_norm = [s.lower().strip() for s in skills1]
    skills2_norm = [s.lower().strip() for s in skills2]
    
    # 1. Similarité lexicale (basée sur les mots exacts)
    set1 = set(skills1_norm)
    set2 = set(skills2_norm)
    
    # Calcul des métriques lexicales
    intersection = len(set1.intersection(set2))
    
    # Jaccard
    union = len(set1.union(set2))
    jaccard = (intersection / union) * 100 if union > 0 else 0
    
    # Chevauchement (Overlap coefficient)
    min_size = min(len(set1), len(set2))
    overlap = (intersection / min_size) * 100 if min_size > 0 else 0
    
    # 2. Similarité sémantique basée sur les embeddings
    semantic_similarity = calculate_skills_similarity_semantic(skills1_norm, skills2_norm)
    
    # 3. Combiner les scores (pondération personnalisable)
    # Donner plus de poids à la similarité sémantique (60%) qu'à la similarité lexicale (40%)
    final_score = (semantic_similarity * 0.6) + (jaccard * 0.2) + (overlap * 0.2)
    
    logger.info(f"Similarité: Sémantique={semantic_similarity:.2f}, Jaccard={jaccard:.2f}, Overlap={overlap:.2f}, Final={final_score:.2f}")
    
    return final_score

def save_cache(cache_file='embedding_cache.pkl'):
    """Sauvegarder le cache des embeddings sur le disque"""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding_cache, f)
        logger.info(f"Cache d'embeddings sauvegardé ({len(embedding_cache)} entrées)")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du cache: {str(e)}")

def load_cache(cache_file='embedding_cache.pkl'):
    """Charger le cache des embeddings depuis le disque"""
    global embedding_cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                embedding_cache = pickle.load(f)
            logger.info(f"Cache d'embeddings chargé ({len(embedding_cache)} entrées)")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du cache: {str(e)}")
            embedding_cache = {}
    else:
        logger.info("Aucun cache d'embeddings trouvé, utilisation d'un cache vide")
        embedding_cache = {} 