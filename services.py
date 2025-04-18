import re
import PyPDF2
import docx2txt
from io import BytesIO

def process_document(file):
    """Extraire le texte d'un fichier PDF ou DOCX"""
    filename = file.filename.lower()
    content = file.read()
    file_stream = BytesIO(content)
    
    if filename.endswith('.pdf'):
        return extract_from_pdf(file_stream)
    elif filename.endswith('.docx'):
        return extract_from_docx(file_stream)
    else:
        return content.decode('utf-8', errors='ignore')

def extract_from_pdf(file_stream):
    """Extraire le texte d'un fichier PDF"""
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_from_docx(file_stream):
    """Extraire le texte d'un fichier DOCX"""
    return docx2txt.process(file_stream)

def clean_text(text):
    """Nettoyer un texte brut"""
    # Supprimer les caractères spéciaux et mettre en forme
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def remove_sensitive_data(text):
    """Supprimer les données sensibles du texte"""
    # Supprimer emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Supprimer numéros de téléphone
    text = re.sub(r'\b(\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b', '[TÉLÉPHONE]', text)
    # Supprimer adresses
    text = re.sub(r'\b\d+\s+[A-Za-z\s]+,\s+[A-Za-z\s]+,\s+[A-Za-z]{2}\s+\d{5}\b', '[ADRESSE]', text)
    return text

def summarize_skills(text):
    """Extraire et résumer les compétences du texte"""
    skills = []
    common_skills = [
        "python", "java", "javascript", "html", "css", "sql", "php", "c#", "c++",
        "react", "angular", "vue", "node", "django", "flask", "spring", "laravel",
        "docker", "kubernetes", "aws", "azure", "gcp", "git", "agile", "scrum",
        "mongodb", "postgresql", "mysql", "oracle", "nosql", "data analysis"
    ]
    
    for skill in common_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            skills.append(skill)
    
    return ", ".join(skills)

def detect_experience_level(text):
    """Détecter le niveau d'expérience"""
    years_exp = re.search(r'\b(\d+)[\s-]*an', text.lower())
    
    if years_exp:
        years = int(years_exp.group(1))
        if years < 2:
            return "Junior"
        elif years < 5:
            return "Intermédiaire"
        else:
            return "Senior"
    
    if re.search(r'\bsenior\b|\bexpert\b|\bchef\b|\blead\b', text.lower()):
        return "Senior"
    elif re.search(r'\bjunior\b|\bdébutant\b|\bstagiaire\b', text.lower()):
        return "Junior"
    
    return "Non spécifié"

def get_embedding(text):
    """Obtenir l'embedding d'un texte (simulé ici)"""
    # Normalement, on utiliserait une bibliothèque comme transformers, sentence-transformers, etc.
    # Ici on simule simplement un vecteur
    import hashlib
    import numpy as np
    
    hash_object = hashlib.md5(text.encode())
    hex_dig = hash_object.hexdigest()
    
    # Crée un vecteur pseudo-aléatoire basé sur le hash du texte
    seed = int(hex_dig, 16) % (2**32)
    np.random.seed(seed)
    
    return np.random.rand(768)  # Dimension courante pour les embeddings