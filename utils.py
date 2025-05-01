import re
import logging
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """Nettoyer un texte brut"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def remove_sensitive_data(text):
    """Supprimer les données sensibles du texte"""
    if not text:
        return ""
    # Supprimer emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Supprimer numéros de téléphone
    text = re.sub(r'\b(\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b', '[TÉLÉPHONE]', text)
    # Supprimer adresses
    text = re.sub(r'\b\d+\s+[A-Za-z\s]+,\s+[A-Za-z\s]+,\s+[A-Za-z]{2}\s+\d{5}\b', '[ADRESSE]', text)
    return text

def extract_phone_numbers(text):
    """Extraire les numéros de téléphone du texte"""
    if not text:
        return None
        
    patterns = [
        r'(?:\+|\(\+)\s*(?:\d{1,4})\s*(?:\)|\s|-|\.)*\s*\d{2,3}(?:\s|-|\.)*\d{2,3}(?:\s|-|\.)*\d{2,3}(?:\s|-|\.)*\d{2,3}',
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b',
        r'\b0\d(?:[-.\s]?\d{2}){4}\b',
        r'\b\d{2}(?:[-.\s]?\d{2}){4}\b',
        r'\b\d{10,12}\b',
        r'\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\(\+\d{1,3}\)[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            phone = matches[0]
            phone = re.sub(r'[^\d+]', '', phone)
            
            if phone.startswith('+'):
                return phone
            elif phone.startswith('0') and len(phone) >= 10:
                return '+33' + phone[1:]
            elif len(phone) > 10 and not phone.startswith('0'):
                return '+' + phone
            else:
                return phone
    
    return None

def extract_email(text):
    """Extraire l'email du texte"""
    if not text:
        return None
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, text)
    return email_matches[0] if email_matches else None 