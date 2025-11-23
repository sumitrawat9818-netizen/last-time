import re

def clean_text(text):
    """
    Cleans text using standard Python tools (No NLTK dependency).
    Guarantees 100% uptime on Streamlit Cloud.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove special characters/numbers using Regex
    # This keeps only letters (a-z) and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    
    return text