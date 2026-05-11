from typing import List, Optional

def preprocess_clinical_input(selected_symptoms: List[str], full_symptoms_list: List[str]) -> List[int]:
    """
    Converts a list of selected symptoms into a binary vector based on the full symptom list.
    
    Example: 
    Inputs: 
        selected_symptoms: ["Cough", "Fever"]
        full_symptoms_list: ["Cough", "Fever", "Dyspnea", "Headache"]
    Output: 
        [1, 1, 0, 0]
    """
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in full_symptoms_list]
    return input_vector

def parse_comma_symptoms(symptoms_str: str) -> List[str]:
    """Parses a comma-separated string for symptom lists."""
    if not symptoms_str or symptoms_str.lower() == "none" or not symptoms_str.strip():
        return []
    return [s.strip() for s in symptoms_str.split(",") if s.strip()]
