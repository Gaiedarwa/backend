"""
Metrics module for calculating precision, accuracy, recall and loss for CV matching
"""
import numpy as np

def calculate_matching_metrics(candidate_skills, job_requirements, matched_skills):
    """
    Calculate precision, recall, F1-score, accuracy and loss for skill matching
    
    Args:
        candidate_skills (list): List of skills from the candidate's CV
        job_requirements (list): List of required skills from the job description
        matched_skills (list): List of skills that matched between CV and job
        
    Returns:
        dict: Dictionary containing various matching metrics
    """
    if not candidate_skills or not job_requirements:
        return {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "accuracy": 0,
            "loss": 1.0
        }
    
    # Normalize all to lowercase for comparison
    candidate_skills_lower = [s.lower() for s in candidate_skills]
    job_requirements_lower = [s.lower() for s in job_requirements]
    matched_skills_lower = [s.lower() for s in matched_skills]
    
    # Calculate true positives, false positives, and false negatives
    true_positives = len(matched_skills_lower)  # Skills that matched
    
    # False positives: skills incorrectly matched
    false_positives = sum(1 for skill in matched_skills_lower if skill not in job_requirements_lower)
    
    # False negatives: required skills missing from matches
    false_negatives = sum(1 for skill in job_requirements_lower if skill not in matched_skills_lower)
    
    # True negatives: skills correctly not matched (skills not in CV and not required by job)
    # This is harder to define precisely, but we can approximate:
    all_unique_skills = set(candidate_skills_lower).union(set(job_requirements_lower))
    true_negatives = len(all_unique_skills) - (true_positives + false_positives + false_negatives)
    
    # Calculate precision: what percentage of matched skills were actually required
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Calculate recall: what percentage of required skills were matched
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score: harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy: percentage of correct predictions (true positives + true negatives)
    total = true_positives + false_positives + false_negatives + true_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    # Calculate loss using cross-entropy inspired approach
    # Higher loss when important skills are missing
    if job_requirements_lower:
        matched_ratio = len(matched_skills_lower) / len(job_requirements_lower)
        # Loss is higher when fewer job requirements are matched
        loss = 1.0 - matched_ratio
    else:
        loss = 0.0
        
    # Apply weighting to prioritize matching critical skills
    critical_skill_count = 0
    matched_critical_count = 0
    
    # Some skills might be more important than others (e.g., programming languages)
    critical_categories = ["programming", "language", "framework", "database", "cloud"]
    
    for skill in job_requirements_lower:
        if any(category in skill for category in critical_categories):
            critical_skill_count += 1
            if skill in matched_skills_lower:
                matched_critical_count += 1
    
    # If there are critical skills, adjust metrics accordingly
    if critical_skill_count > 0:
        critical_recall = matched_critical_count / critical_skill_count
        # Blend regular recall with critical recall (60% weight to critical skills)
        adjusted_recall = (recall * 0.4) + (critical_recall * 0.6)
        
        # Adjust loss to penalize missing critical skills more heavily
        critical_loss = 1.0 - critical_recall
        loss = (loss * 0.4) + (critical_loss * 0.6)
        
        # Update other metrics
        if precision > 0:
            f1_score = 2 * (precision * adjusted_recall) / (precision + adjusted_recall)
    
    # Ensure all values are between 0 and 1, then convert to percentages
    metrics = {
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1_score": round(f1_score * 100, 1),
        "accuracy": round(accuracy * 100, 1),
        "loss": round(loss, 3)
    }
    
    return metrics 