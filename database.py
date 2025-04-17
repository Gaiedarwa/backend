import pymongo
from pymongo import MongoClient
import logging
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis un fichier .env (s'il existe)
load_dotenv()

# Configuration de la connexion MongoDB
try:
    # Obtenir l'URL de connexion depuis les variables d'environnement ou utiliser une valeur par défaut
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    
    # Sélection de la base de données
    db = client.recrutement_app
    
    # Initialisation des collections
    cv_collection = db.cvs
    offers_collection = db.offers
    
    # Création d'index pour améliorer les performances
    cv_collection.create_index([("offer_id", pymongo.ASCENDING)])
    cv_collection.create_index([("created_at", pymongo.DESCENDING)])
    offers_collection.create_index([("created_at", pymongo.DESCENDING)])
    offers_collection.create_index([("keywords", pymongo.TEXT)])
    
    # Tester la connexion
    client.admin.command('ping')
    logging.info("✅ Connexion à MongoDB établie avec succès!")
    
except Exception as e:
    logging.error(f"❌ Erreur de connexion à MongoDB: {e}")
    # Créer des collections fictives pour le développement
    from pymongo.errors import ServerSelectionTimeoutError
    
    class MockCollection:
        def __init__(self, name):
            self.name = name
            self.data = []
            self._id_counter = 1
        
        def insert_one(self, document):
            document["_id"] = self._id_counter
            self._id_counter += 1
            self.data.append(document)
            
            class InsertOneResult:
                def __init__(self, inserted_id):
                    self.inserted_id = inserted_id
            
            return InsertOneResult(document["_id"])
        
        def find(self, query=None):
            return self.data
        
        def find_one(self, query=None):
            if not self.data:
                return None
            return self.data[0]
        
        def create_index(self, keys, **kwargs):
            pass
    
    cv_collection = MockCollection("cvs")
    offers_collection = MockCollection("offers")
    logging.warning("⚠️ Utilisation de collections fictives pour le développement") 