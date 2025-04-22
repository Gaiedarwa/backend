import os
import shutil
import sys
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("migration")

# Fichiers à supprimer (ancienne architecture)
files_to_remove = [
    "routes.py",
    "services.py",
    "models.py",
    "embeddings.py",
    "summarization.py",
    "skills.py",
    "accuracy.py",
    "tech_terms.py",
    "test_generator.py",
    "technical_test.py",
    "metrics.py",
]

def confirm_migration():
    """Demande confirmation à l'utilisateur avant de procéder à la migration"""
    print("\n⚠️ ATTENTION: Ce script va supprimer les anciens fichiers de l'architecture.")
    print("Assurez-vous d'avoir fait une sauvegarde avant de continuer.")
    print("Les fichiers suivants seront supprimés:")
    for file in files_to_remove:
        if os.path.exists(file):
            print(f" - {file}")
    
    confirmation = input("\nÊtes-vous sûr de vouloir continuer? (oui/non): ")
    return confirmation.lower() in ['oui', 'o', 'yes', 'y']

def remove_old_files():
    """Supprime les anciens fichiers"""
    success_count = 0
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"✅ Fichier supprimé: {file}")
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Erreur lors de la suppression de {file}: {str(e)}")
        else:
            logger.warning(f"⚠️ Fichier non trouvé: {file}")
    
    return success_count

def verify_new_architecture():
    """Vérifie que les fichiers nécessaires existent dans la nouvelle architecture"""
    required_files = [
        "app/__init__.py",
        "app/database/__init__.py",
        "app/models/summarization.py",
        "app/models/skills.py",
        "app/models/similarity.py",
        "app/routes/job_offers.py",
        "app/routes/applications.py",
        "app/services/job_offer_service.py",
        "app/services/application_service.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def main():
    """Fonction principale pour exécuter la migration"""
    logger.info("📂 Démarrage de la migration vers la nouvelle architecture")
    
    # Vérifier la nouvelle architecture
    missing_files = verify_new_architecture()
    if missing_files:
        logger.error("❌ Des fichiers essentiels de la nouvelle architecture sont manquants:")
        for file in missing_files:
            logger.error(f" - {file}")
        logger.error("Migration annulée. Veuillez vérifier la nouvelle architecture.")
        return
    
    # Demander confirmation
    if not confirm_migration():
        logger.info("Migration annulée par l'utilisateur.")
        return
    
    # Supprimer les anciens fichiers
    success_count = remove_old_files()
    
    logger.info(f"✅ Migration terminée! {success_count} fichiers supprimés.")
    logger.info("Pour lancer l'application avec la nouvelle architecture, exécutez:")
    logger.info("python app.py")

if __name__ == "__main__":
    main() 