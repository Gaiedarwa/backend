import re
import logging
from models import extract_entities
from sklearn.metrics import precision_score, recall_score, f1_score

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
    
    # Évaluer l'éducation
    gt_education = set([e.lower() for e in ground_truth.get('education', [])])
    ex_education = set([e.lower() for e in extracted.get('education', [])])
    
    # Pour l'éducation, on utilise une correspondance plus souple
    education_matched = 0
    for gt_edu in gt_education:
        for ex_edu in ex_education:
            # Vérifier si au moins 50% des mots du gt_edu se trouvent dans ex_edu
            gt_words = set(gt_edu.split())
            ex_words = set(ex_edu.split())
            common_words = gt_words.intersection(ex_words)
            
            if len(common_words) >= len(gt_words) * 0.5:
                education_matched += 1
                break
    
    education_precision = education_matched / len(ex_education) if len(ex_education) > 0 else 0
    education_recall = education_matched / len(gt_education) if len(gt_education) > 0 else 0
    education_f1 = 2 * (education_precision * education_recall) / (education_precision + education_recall) if (education_precision + education_recall) > 0 else 0
    
    results['education'] = {
        'precision': education_precision,
        'recall': education_recall,
        'f1': education_f1
    }
    
    # Calculer le score global
    results['overall'] = {
        'precision': (personal_precision + skills_precision + education_precision) / 3,
        'recall': (1.0 + skills_recall + education_recall) / 3,  # Supposons recall=1.0 pour les infos personnelles
        'f1': (1.0 + skills_f1 + education_f1) / 3,  # Supposons f1=1.0 pour les infos personnelles
        'accuracy': personal_correct / personal_total if personal_total > 0 else 0
    }
    
    return results

def batch_evaluate(cv_texts, ground_truths):
    """
    Évaluer la précision sur un ensemble de CV
    
    Args:
        cv_texts (list): Liste de textes de CV
        ground_truths (list): Liste de dictionnaires contenant les vérités de terrain
        
    Returns:
        dict: Métriques d'évaluation moyennes
    """
    if len(cv_texts) != len(ground_truths):
        raise ValueError("Le nombre de CV et de vérités de terrain doit être identique")
    
    all_results = []
    for i, (cv, gt) in enumerate(zip(cv_texts, ground_truths)):
        result = evaluate_extraction_accuracy(cv, gt)
        all_results.append(result)
        logging.info(f"CV {i+1}/{len(cv_texts)}: Précision globale = {result['overall']['precision']:.2f}")
    
    # Calculer les moyennes
    mean_results = {
        'overall': {
            'precision': sum(r['overall']['precision'] for r in all_results) / len(all_results),
            'recall': sum(r['overall']['recall'] for r in all_results) / len(all_results),
            'f1': sum(r['overall']['f1'] for r in all_results) / len(all_results),
            'accuracy': sum(r['overall']['accuracy'] for r in all_results) / len(all_results)
        },
        'personal_info': {
            'precision': sum(r['personal_info']['precision'] for r in all_results) / len(all_results)
        },
        'skills': {
            'precision': sum(r['skills']['precision'] for r in all_results) / len(all_results),
            'recall': sum(r['skills']['recall'] for r in all_results) / len(all_results),
            'f1': sum(r['skills']['f1'] for r in all_results) / len(all_results)
        },
        'education': {
            'precision': sum(r['education']['precision'] for r in all_results) / len(all_results),
            'recall': sum(r['education']['recall'] for r in all_results) / len(all_results),
            'f1': sum(r['education']['f1'] for r in all_results) / len(all_results)
        }
    }
    
    return mean_results

# Route d'API pour évaluer la précision
def create_accuracy_routes(app):
    from flask import request, jsonify
    
    @app.route('/evaluate/accuracy', methods=['POST'])
    def evaluate_accuracy():
        """
        Endpoint pour évaluer la précision du modèle d'extraction
        
        Requête:
        {
            "cv_text": "Texte brut du CV",
            "ground_truth": {
                "name": "Nom complet",
                "email": "email@example.com",
                "telephone": "0123456789",
                "skills": ["python", "java", "machine learning"],
                "education": ["Master en informatique"],
                "experience": ["5 ans d'expérience en développement"]
            }
        }
        
        Réponse:
        {
            "overall": {
                "precision": 0.85,
                "recall": 0.78,
                "f1": 0.81,
                "accuracy": 0.75
            },
            "personal_info": {...},
            "skills": {...},
            "education": {...},
            "experience": {...}
        }
        """
        try:
            data = request.json
            cv_text = data.get('cv_text')
            ground_truth = data.get('ground_truth')
            
            if not cv_text or not ground_truth:
                return jsonify({"error": "Les champs 'cv_text' et 'ground_truth' sont requis"}), 400
            
            result = evaluate_extraction_accuracy(cv_text, ground_truth)
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Erreur lors de l'évaluation de la précision: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/evaluate/batch', methods=['POST'])
    def evaluate_batch():
        """
        Endpoint pour évaluer la précision sur un lot de CV
        
        Requête:
        {
            "cv_texts": ["Texte CV 1", "Texte CV 2", ...],
            "ground_truths": [
                {
                    "name": "Nom 1",
                    "email": "email1@example.com",
                    ...
                },
                {
                    "name": "Nom 2",
                    "email": "email2@example.com",
                    ...
                },
                ...
            ]
        }
        
        Réponse:
        {
            "overall": {
                "precision": 0.82,
                "recall": 0.75,
                "f1": 0.78,
                "accuracy": 0.70
            },
            ...
        }
        """
        try:
            data = request.json
            cv_texts = data.get('cv_texts')
            ground_truths = data.get('ground_truths')
            
            if not cv_texts or not ground_truths:
                return jsonify({"error": "Les champs 'cv_texts' et 'ground_truths' sont requis"}), 400
            
            result = batch_evaluate(cv_texts, ground_truths)
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Erreur lors de l'évaluation par lot: {str(e)}")
            return jsonify({"error": str(e)}), 500 