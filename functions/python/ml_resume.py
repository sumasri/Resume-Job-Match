import os
import PyPDF2
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
import openai  # OpenAI API integration

# ---------------------------------------------------------
# Initialize Models
# ---------------------------------------------------------
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# OpenAI API Key (ensure it's set in your environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "openai" 

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = "".join(page.extract_text() for page in reader.pages)
        print(f"[INFO] Extracted text from PDF: {pdf_path}")
        return text
    except PyPDF2.errors.PdfReadError as e:
        print(f"[ERROR] Could not read PDF file {pdf_path}: {e}")
        return ""

def read_text_file(file_path):
    """Reads text content from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"[INFO] Extracted text from text file: {file_path}")
        return content
    except Exception as e:
        print(f"[ERROR] Could not read text file {file_path}: {e}")
        return ""

def extract_keywords(text):
    """Extracts keywords from text using SpaCy."""
    doc = nlp(text)
    keywords = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 1]
    print(f"[INFO] Extracted Keywords: {keywords[:10]}...")
    return keywords

def enhance_keywords(keywords):
    """Enhances keywords using OpenAI's GPT model."""
    prompt = f"Extracted keywords: {', '.join(keywords)}\nSuggest additional related skills or synonyms."
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant helping to enhance extracted keywords."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        content = response.choices[0].message.content
        enhanced_keywords = [keyword.strip() for keyword in content.split('\n') if keyword.strip()]
        print(f"[INFO] Enhanced Keywords: {enhanced_keywords[:10]}...")
        return enhanced_keywords
    except Exception as e:
        print(f"[ERROR] Failed to enhance keywords: {e}")
        return keywords  # Fallback to original keywords if enhancement fails

def keyword_match_score(text1, text2):
    """Calculates keyword match score using TF-IDF."""
    if not text1.strip() or not text2.strip():
        print("[WARN] One or both inputs to keyword_match_score are empty.")
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    print(f"[INFO] Keyword Match Score: {score:.2f}")
    return score

def semantic_similarity(text1, text2):
    """Calculates semantic similarity between two texts."""
    if not text1.strip() or not text2.strip():
        print("[WARN] One or both inputs to semantic_similarity are empty.")
        return 0.0
    embeddings1 = similarity_model.encode(text1, convert_to_tensor=True)
    embeddings2 = similarity_model.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    print(f"[INFO] Semantic Similarity Score: {score:.2f}")
    return score

def extract_section(text, section_name):
    """Extracts specific sections (e.g., job title, skills) from text using QA model."""
    try:
        result = qa_model(question=f"Extract {section_name} information.", context=text)
        extracted_section = result['answer']
        print(f"[INFO] Extracted {section_name}: {extracted_section[:50]}...")
        return extracted_section
    except Exception as e:
        print(f"[WARN] Failed to extract {section_name}: {e}")
        return ""

# ---------------------------------------------------------
# Feature Calculation
# ---------------------------------------------------------

def calculate_features(resume_text, job_desc_text):
    """Calculates features for ML model."""
    # Extract keywords and enhance them
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_desc_text)
    enhanced_resume_keywords = enhance_keywords(resume_keywords)
    enhanced_job_keywords = enhance_keywords(job_keywords)

    # Calculate overall match score
    overall_score = keyword_match_score(" ".join(enhanced_resume_keywords), " ".join(enhanced_job_keywords))

    # Extract specific category scores
    job_title_score = semantic_similarity(
        extract_section(job_desc_text, "job title"),
        extract_section(resume_text, "job title")
    ) * 100

    skills_score = keyword_match_score(
        extract_section(job_desc_text, "skills"),
        extract_section(resume_text, "skills")
    )

    experience_score = semantic_similarity(
        extract_section(job_desc_text, "experience"),
        extract_section(resume_text, "experience")
    ) * 100

    education_score = semantic_similarity(
        extract_section(job_desc_text, "education"),
        extract_section(resume_text, "education")
    ) * 100

    print(f"[INFO] Calculated Features: Overall: {overall_score:.2f}, Job Title: {job_title_score:.2f}, "
          f"Skills: {skills_score:.2f}, Experience: {experience_score:.2f}, Education: {education_score:.2f}")
    return [overall_score, job_title_score, skills_score, experience_score, education_score]

# ---------------------------------------------------------
# Machine Learning Functions
# ---------------------------------------------------------

def train_model(resume_dir, job_desc_text):
    """Trains an ML model using resumes from the specified directory."""
    X = []
    y = []

    for resume_file in os.listdir(resume_dir):
        resume_path = os.path.join(resume_dir, resume_file)
        if resume_path.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_path)
            features = calculate_features(resume_text, job_desc_text)
            X.append(features)
            y.append(np.random.randint(50, 100))  # Simulate labels; replace with real labels for better accuracy

    print("[DEBUG] Training Features (X):", X)
    print("[DEBUG] Training Labels (y):", y)

    X = np.array(X)
    y = np.array(y)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model
    with open("resume_match_model.pkl", "wb") as file:
        pickle.dump(model, file)

    print(f"[INFO] Trained model saved with {len(X)} samples.")
    return model

def load_model():
    """Loads a pre-trained ML model."""
    try:
        with open("resume_match_model.pkl", "rb") as file:
            model = pickle.load(file)
        print("[INFO] Pre-trained model loaded successfully.")
        return model
    except FileNotFoundError:
        print("[ERROR] Model file not found. Please train the model first.")
        return None

def predict_match_score(model, resume_text, job_desc_text):
    """Predicts the match score for a given resume and job description."""
    features = calculate_features(resume_text, job_desc_text)
    prediction = model.predict([features])[0]
    print(f"[INFO] Predicted Match Score: {prediction:.2f}")
    return prediction

# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------

def main():
    """Main entry point for the script."""
    engineering_dir = "data/data/ENGINEERING"  # Directory containing ENGINEERING resumes
    job_desc_path = "job-desc.txt"  # Path to the target job description
    target_resume_path = "10030015.pdf"  # Path to the target resume

    # Extract job description
    job_desc_text = read_text_file(job_desc_path) if job_desc_path.endswith(".txt") else extract_text_from_pdf(job_desc_path)

    # Train or load the ML model
    model = load_model()
    if not model:
        print("Training model on ENGINEERING resumes...")
        model = train_model(engineering_dir, job_desc_text)

    # Predict match score for the target resume
    target_resume_text = extract_text_from_pdf(target_resume_path)
    if target_resume_text:
        match_score = predict_match_score(model, target_resume_text, job_desc_text)
        print(f"\nPredicted Match Score for Target Resume: {match_score:.2f}/100")
    else:
        print("[ERROR] Failed to extract text from the target resume.")

if __name__ == "__main__":
    main()
