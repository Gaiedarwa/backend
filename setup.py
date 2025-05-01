import nltk
import subprocess
import sys

def install_requirements():
    """Install required packages from requirements.txt"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_resources():
    """Download required NLTK resources"""
    nltk.download('punkt')
    nltk.download('stopwords')

if __name__ == "__main__":
    print("Installing required packages...")
    install_requirements()
    print("Downloading NLTK resources...")
    download_nltk_resources()
    print("Setup completed successfully!") 