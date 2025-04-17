import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import logging

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Liste de compétences techniques courantes
COMMON_SKILLS = [
    # Langages de programmation
    "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", "swift", "kotlin",
    "scala", "perl", "golang", "r", "groovy", "objective-c", "rust", "dart", "haskell", "lua",
    
    # Web et frontend
    "html", "css", "react", "angular", "vue", "jquery", "bootstrap", "sass", "less", "webpack",
    "redux", "graphql", "next.js", "gatsby", "nuxt.js", "svelte", "ember", "ionic", "pwa",
    
    # Backend et frameworks
    "node", "django", "flask", "spring", "laravel", "symfony", "express", "fastapi", "rails",
    "asp.net", "hibernate", "struts", "play", "phoenix", "gin", "quarkus", "micronaut",
    
    # Base de données
    "sql", "mongodb", "postgresql", "mysql", "sqlite", "oracle", "cassandra", "redis", "dynamodb",
    "couchdb", "neo4j", "firebase", "elasticsearch", "mariadb", "influxdb", "cosmos db",
    
    # DevOps et Cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible", "jenkins", "circleci",
    "travis", "github actions", "gitlab ci", "puppet", "chef", "nagios", "prometheus", "grafana",
    
    # Méthodologies et pratiques
    "agile", "scrum", "kanban", "lean", "waterfall", "devops", "ci/cd", "tdd", "bdd", "sre",
    
    # IA et science des données
    "machine learning", "deep learning", "ai", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "nlp", "computer vision", "neural networks",
    
    # Systèmes d'exploitation et environnements
    "linux", "windows", "macos", "ios", "android", "unix", "bash", "powershell", "embedded systems",
    
    # Autres technologies et concepts
    "rest api", "soap", "graphql", "microservices", "serverless", "blockchain", "iot",
    "augmented reality", "virtual reality", "big data", "etl", "web services", "web sockets"
]

# Mots-clés non pertinents à filtrer des compétences
NON_SKILL_KEYWORDS = [
    "de", "et", "le", "la", "les", "du", "des", "un", "une", "en", "à", "au", "aux", "pour", "par", "sur",
    "dans", "avec", "sans", "sous", "entre", "vers", "chez", "dès", "of", "the", "and", "or", "for", "to",
    "from", "with", "without", "by", "at", "in", "on", "under", "between", "education", "experience", 
    "certification", "formation", "expérience", "institut", "university", "université", "lycée", "school",
    "école", "profile", "profil", "contact", "email", "téléphone", "phone", "address", "adresse", "name",
    "nom", "prénom", "age", "birth", "naissance", "degree", "studies", "étude", "études", "année", "year",
    "date", "month", "mois", "day", "jour", "certificate", "certificat", "diploma", "diplôme", "bachelor",
    "master", "phd", "doctorat", "engineering", "ingénieur", "stage", "internship", "stagiaire", "intern",
    "project", "projet", "work", "travail", "job", "emploi", "company", "entreprise", "organization",
    "organisation", "department", "département", "service"
]

def extract_entities(text):
    """
    Extraire des entités nommées du texte avec une méthode améliorée
    
    Args:
        text (str): Texte brut du CV
    
    Returns:
        dict: Dictionnaire contenant les entités extraites
    """
    entities = {
        'skills': [],
        'education': [],
        'experience': [],
        'name': None,
        'email': None,
        'telephone': None
    }
    
    # Extraction de l'email (expression régulière plus précise)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, text)
    if email_matches:
        entities['email'] = email_matches[0]  # Prendre le premier email trouvé
    
    # Extraction du téléphone (expressions régulières pour différents formats)
    phone_patterns = [
        r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Format international: +1 (123) 456-7890
        r'\b\d{8,10}\b',  # Format simple: 1234567890
        r'\b\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b',  # Format français: 06 12 34 56 78
        r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b'  # Format alternatif: 123 45 67 89
    ]
    
    for pattern in phone_patterns:
        phone_matches = re.findall(pattern, text)
        if phone_matches:
            # Nettoyer et formater le numéro de téléphone
            phone = phone_matches[0]
            # Supprimer les espaces, tirets, etc.
            phone = re.sub(r'[^\d+]', '', phone)
            # Ajouter un formatage standard
            if len(phone) >= 10:
                if not phone.startswith('+'):
                    if phone.startswith('00'):
                        phone = '+' + phone[2:]
                    elif len(phone) == 10:  # Numéro français sans indicatif
                        phone = '+33' + phone[1:] if phone.startswith('0') else '+33' + phone
                entities["telephone"] = phone
            break
    
    # Extraction du nom (plusieurs approches)
    # 1. Chercher dans les premières lignes
    first_lines = text.strip().split('\n')[:5]  # Examiner les 5 premières lignes
    
    for line in first_lines:
        # Motif pour un nom complet (1-2 mots, première lettre majuscule)
        name_match = re.search(r'^([A-Z][a-zàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ]{1,}[\s-]?)+$', line.strip())
        if name_match and len(line.strip()) > 3 and len(line.strip()) < 50:
            entities['name'] = line.strip()
            break
    
    # Extraction améliorée des compétences
    text_lower = text.lower()
    extracted_skills = set()
    
    # 1. Rechercher les compétences communes par regex
    for skill in COMMON_SKILLS:
        # Utiliser une regex pour trouver des mots entiers
        skill_pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(skill_pattern, text_lower):
            extracted_skills.add(skill)
    
    # 2. Rechercher des sections de compétences dans le CV
    skill_section_patterns = [
        r'(?:compétences|skills|technical skills|competencies|compétences techniques)(?:[\s\n:]+)([\s\S]*?)(?:\n\n|\n\w|$)',
        r'(?:technologies|outils|tools|frameworks|langages|languages)(?:[\s\n:]+)([\s\S]*?)(?:\n\n|\n\w|$)'
    ]
    
    for pattern in skill_section_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            for skill_section in matches:
                # Diviser la section en éléments potentiels de compétence
                skill_items = re.split(r'[,;•\n]+', skill_section)
                
                for skill_item in skill_items:
                    skill_item = skill_item.strip()
                    
                    # Vérifier si c'est une compétence valide (pas trop court, pas trop long)
                    if skill_item and 2 < len(skill_item) < 30:
                        # Vérifier si ce n'est pas un mot non pertinent
                        if skill_item not in NON_SKILL_KEYWORDS:
                            # Vérifier si c'est une compétence connue ou si cela ressemble à une compétence technique
                            if skill_item in COMMON_SKILLS or re.match(r'^[a-z0-9\-\.]+$', skill_item):
                                extracted_skills.add(skill_item)
    
    # 3. Rechercher des compétences dans les sections d'expérience
    experience_section_pattern = r'(?:expérience|experience|professional experience|parcours professionnel|work)(?:[\s\n:]+)([\s\S]*?)(?:\n\n|\n\w|$)'
    exp_matches = re.findall(experience_section_pattern, text_lower)
    
    if exp_matches:
        for exp_section in exp_matches:
            # Rechercher des mots qui ressemblent à des technologies dans cette section
            for skill in COMMON_SKILLS:
                skill_pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(skill_pattern, exp_section):
                    extracted_skills.add(skill)
            
            # Rechercher des motifs comme "Technologies: X, Y, Z"
            tech_pattern = r'(?:technologies|tech stack|stack|techno|outils|tools)[\s:]+([^\n\.]+)'
            tech_matches = re.findall(tech_pattern, exp_section)
            
            for tech_list in tech_matches:
                skills_from_list = re.split(r'[,;/]+', tech_list)
                for skill in skills_from_list:
                    skill = skill.strip()
                    if skill and 2 < len(skill) < 30 and skill not in NON_SKILL_KEYWORDS:
                        extracted_skills.add(skill)
    
    # Filtrer les compétences trop génériques ou qui ne sont pas réellement des compétences
    filtered_skills = []
    for skill in extracted_skills:
        # Vérifier si c'est une phrase et non une compétence
        if len(skill.split()) > 3:
            continue
            
        # Vérifier s'il contient des données qui ne devraient pas être des compétences
        if any(non_skill in skill for non_skill in NON_SKILL_KEYWORDS):
            continue
            
        # Vérifier si le mot contient des caractères suspects
        if re.search(r'[^\w\s\-\.]', skill):
            continue
            
        filtered_skills.append(skill)
    
    entities['skills'] = filtered_skills
    
    # Extraction de l'éducation
    edu_section_pattern = r'(?:éducation|education|formation|études|studies|parcours scolaire|academic)(?:[\s\n:]+)([\s\S]*?)(?:\n\n|\n\w|$)'
    edu_matches = re.findall(edu_section_pattern, text_lower)
    
    if edu_matches:
        for edu_section in edu_matches:
            # Diviser en paragraphes
            edu_paragraphs = re.split(r'\n+', edu_section)
            for para in edu_paragraphs:
                para = para.strip()
                if para and len(para) > 10:
                    # Exclure les paragraphes qui contiennent clairement des informations personnelles
                    if not re.search(email_pattern, para) and not re.search(r'\b\d{5,}\b', para):
                        entities['education'].append(para)
    
    # Extraction de l'expérience professionnelle
    exp_section_pattern = r'(?:expérience|experience|professional experience|parcours professionnel|work)(?:[\s\n:]+)([\s\S]*?)(?:\n\n|\n\w|$)'
    exp_matches = re.findall(exp_section_pattern, text_lower)
    
    if exp_matches:
        for exp_section in exp_matches:
            # Diviser en paragraphes
            exp_paragraphs = re.split(r'\n\n+', exp_section)
            for para in exp_paragraphs:
                para = para.strip()
                if para and len(para) > 10:
                    # Nettoyer l'expérience pour supprimer les données personnelles
                    clean_para = clean_experience_text(para)
                    if clean_para:
                        entities['experience'].append(clean_para)
    
    return entities

def clean_experience_text(text):
    """
    Nettoyer le texte d'expérience pour supprimer les données personnelles
    et ne garder que les informations professionnelles pertinentes
    
    Args:
        text (str): Texte brut d'une expérience
        
    Returns:
        str: Texte nettoyé
    """
    # Supprimer les emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Supprimer les numéros de téléphone
    text = re.sub(r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[TEL]', text)
    text = re.sub(r'\b\d{8,10}\b', '[TEL]', text)
    text = re.sub(r'\b\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b', '[TEL]', text)
    
    # Supprimer les URLs
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'www\.\S+', '[URL]', text)
    
    # Nettoyer les lignes vides multiples
    text = re.sub(r'\n{2,}', '\n', text)
    text = text.strip()
    
    # Limiter la longueur
    if len(text) > 500:
        text = text[:497] + "..."
    
    return text

def validate_personal_info(text):
    """Valider les informations personnelles dans le texte"""
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'\b(\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b', text))
    has_name = bool(re.search(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', text.strip().split('\n')[0]))
    
    return {
        'has_email': has_email,
        'has_phone': has_phone,
        'has_name': has_name,
        'is_valid': has_email and has_phone
    }

def calculate_similarity(embedding1, embedding2):
    """Calculer la similarité cosinus entre deux embeddings"""
    # Convertir en matrices 2D pour cosine_similarity
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    
    similarity = cosine_similarity(emb1, emb2)[0][0]
    
    # Convertir en pourcentage (0-100)
    return float(similarity * 100)

def calculate_skills_similarity(skills1, skills2):
    """
    Calculer la similarité entre deux ensembles de compétences
    
    Args:
        skills1 (list): Premier ensemble de compétences
        skills2 (list): Deuxième ensemble de compétences
    
    Returns:
        float: Score de similarité en pourcentage (0-100)
    """
    # Cette fonction est maintenant remplacée par celle dans embeddings.py
    # On la garde ici pour la compatibilité, mais elle redirigera vers la nouvelle
    try:
        from embeddings import calculate_skills_similarity as semantic_similarity
        return semantic_similarity(skills1, skills2)
    except ImportError:
        # Fallback si le module embeddings n'est pas disponible
        if not skills1 or not skills2:
            return 0
        
        # Normaliser les compétences (minuscules et suppression des espaces)
        skills1_norm = [s.lower().strip() for s in skills1]
        skills2_norm = [s.lower().strip() for s in skills2]
        
        # Convertir les listes en ensembles pour calculer l'intersection
        set1 = set(skills1_norm)
        set2 = set(skills2_norm)
        
        # Jaccard similarité
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard_similarity = (intersection / union) * 100 if union > 0 else 0
        
        # Similarité par chevauchement (overlap coefficient)
        min_size = min(len(set1), len(set2))
        overlap_similarity = (intersection / min_size) * 100 if min_size > 0 else 0
        
        # Augmenter l'importance de la correspondance exacte
        exact_match_bonus = min(30, intersection * 5)  # Bonus maximal de 30%
        
        # Méthode TF-IDF pour les grands ensembles
        tfidf_similarity = 0
        if len(skills1) > 3 and len(skills2) > 3:
            try:
                # Créer des documents à partir des compétences
                doc1 = ' '.join(skills1_norm)
                doc2 = ' '.join(skills2_norm)
                
                # Calculer les scores TF-IDF
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
                
                # Calculer la similarité cosinus
                tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
            except Exception as e:
                logging.warning(f"Erreur lors du calcul TF-IDF: {str(e)}")
        
        # Combiner les méthodes avec pondération
        final_similarity = (jaccard_similarity * 0.4) + (overlap_similarity * 0.3) + (tfidf_similarity * 0.2) + (exact_match_bonus * 0.1)
        
        return final_similarity

def evaluate_extraction_accuracy(cv_text, ground_truth):
    """
    Évaluer la précision de l'extraction d'informations à partir d'un CV
    
    Args:
        cv_text (str): Texte brut du CV
        ground_truth (dict): Valeurs véritables annotées manuellement
            format: {
                'name': str,
                'email': str,
                'telephone': str,
                'skills': list,
                'education': list,
                'experience': list
            }
    
    Returns:
        dict: Métriques d'évaluation (précision, rappel, F1 pour chaque catégorie)
    """
    # Extraire les entités avec notre modèle
    extracted = extract_entities(cv_text)
    
    results = {
        'overall': {},
        'personal_info': {},
        'skills': {},
        'education': {},
        'experience': {}
    }
    
    # Évaluer les informations personnelles
    personal_fields = ['name', 'email', 'telephone']
    personal_correct = 0
    personal_total = 0
    
    for field in personal_fields:
        if ground_truth.get(field) and extracted.get(field):
            # Pour les informations personnelles, on vérifie si l'extrait contient la vérité
            if ground_truth[field].lower() in extracted[field].lower():
                personal_correct += 1
            personal_total += 1
        elif ground_truth.get(field) is None and extracted.get(field) is None:
            # Les deux sont None, c'est correct
            personal_correct += 1
            personal_total += 1
        else:
            # L'un est None et l'autre non, c'est incorrect
            personal_total += 1
    
    personal_precision = personal_correct / personal_total if personal_total > 0 else 0
    results['personal_info'] = {
        'precision': personal_precision,
        'details': {
            field: {
                'ground_truth': ground_truth.get(field),
                'extracted': extracted.get(field),
                'match': ground_truth.get(field) and extracted.get(field) and ground_truth[field].lower() in extracted[field].lower()
            } for field in personal_fields
        }
    }
    
    # Évaluer les compétences (skills)
    gt_skills = set([s.lower() for s in ground_truth.get('skills', [])])
    ex_skills = set([s.lower() for s in extracted.get('skills', [])])
    
    skills_tp = len(gt_skills.intersection(ex_skills))  # True positives
    skills_fp = len(ex_skills - gt_skills)  # False positives
    skills_fn = len(gt_skills - ex_skills)  # False negatives
    
    skills_precision = skills_tp / (skills_tp + skills_fp) if (skills_tp + skills_fp) > 0 else 0
    skills_recall = skills_tp / (skills_tp + skills_fn) if (skills_tp + skills_fn) > 0 else 0
    skills_f1 = 2 * (skills_precision * skills_recall) / (skills_precision + skills_recall) if (skills_precision + skills_recall) > 0 else 0
    
    results['skills'] = {
        'precision': skills_precision,
        'recall': skills_recall,
        'f1': skills_f1,
        'details': {
            'true_positives': list(gt_skills.intersection(ex_skills)),
            'false_positives': list(ex_skills - gt_skills),
            'false_negatives': list(gt_skills - ex_skills)
        }
    }
    
    # Calculer le score global
    results['overall'] = {
        'precision': (personal_precision + skills_precision) / 2,
        'accuracy': (personal_correct / personal_total if personal_total > 0 else 0) * 0.5 + (skills_tp / len(gt_skills) if len(gt_skills) > 0 else 0) * 0.5
    }
    
    return results 