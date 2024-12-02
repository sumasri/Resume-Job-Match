import os
import PyPDF2
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai  # For generative AI

# Initialize models
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# OpenAI API key (set this in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "openai" 

def read_job_description(file_path):
    """Reads the job description from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text


def extract_info(text, question):
    """Extracts specific information from the text using a QA model."""
    result = qa_model(question=question, context=text, max_answer_len=100)
    return result['answer']


def semantic_similarity(text1, text2):
    """Calculates semantic similarity between two texts."""
    embeddings1 = similarity_model.encode(text1, convert_to_tensor=True)
    embeddings2 = similarity_model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return similarity


def extract_keywords(text):
    """Extracts keywords from text using SpaCy."""
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 1]


def enhance_keywords(keywords):
    """Enhances keywords using OpenAI's GPT model."""
    prompt = f"Extracted keywords: {', '.join(keywords)}\nSuggest additional related skills or synonyms."
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use a supported model
        messages=[
            {"role": "system", "content": "You are an assistant helping to suggest additional keywords."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    # Correctly access the content
    content = response.choices[0].message.content  # Corrected access to the content
    return [keyword.strip() for keyword in content.split('\n') if keyword.strip()]


def keyword_match_score(text1, text2):
    """Calculates keyword match score using TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100


def calculate_resume_match(resume_file, job_desc_file):
    """Main function to calculate match score between a resume and job description."""
    # Read and process input files
    job_description = read_job_description(job_desc_file)
    resume_text = extract_text_from_pdf(resume_file)

    # Extract keywords from both documents
    resume_keywords = extract_keywords(resume_text)
    job_desc_keywords = extract_keywords(job_description)

    # Enhance extracted keywords using generative AI
    enhanced_keywords = enhance_keywords(resume_keywords)

    # Calculate overall score based on keyword matches
    overall_score = round(keyword_match_score(' '.join(resume_keywords), ' '.join(job_desc_keywords)))

    # Extract specific information from the documents
    job_title_desc = extract_info(job_description, "What is the job title and main responsibilities?")
    job_title_resume = extract_info(resume_text, "What is the applicant's current job title and main responsibilities?")
    
    skills_desc = extract_info(job_description, "What are the key skills, technologies, and qualifications required for this job?")
    skills_resume = extract_info(resume_text, "What are the applicant's key skills, technologies, and qualifications?")
    
    experience_desc = extract_info(job_description, "What kind of work experience and achievements are required for this job?")
    experience_resume = extract_info(resume_text, "What is the applicant's work experience and key achievements?")
    
    education_desc = extract_info(job_description, "What educational background and certifications are required for this job?")
    education_resume = extract_info(resume_text, "What is the applicant's educational background and certifications?")

    # Calculate individual similarity scores
    job_title_score = round(semantic_similarity(job_title_desc, job_title_resume) * 100)
    skills_score = round(keyword_match_score(skills_desc, skills_resume))
    experience_score = round(semantic_similarity(experience_desc, experience_resume) * 100)
    education_score = round(semantic_similarity(education_desc, education_resume) * 100)

    # Print results
    print(f"Overall Match Score: {overall_score}/100")
    print(f"Enhanced Keywords: {', '.join(enhanced_keywords)}")
    print(f"Job Title Match Score: {job_title_score}/100")
    print(f"Skills Match Score: {skills_score}/100")
    print(f"Experience Match Score: {experience_score}/100")
    print(f"Education Match Score: {education_score}/100")

    return {
        'match_score': overall_score,
        'enhanced_keywords': enhanced_keywords,
        'job_title': job_title_score,
        'skills': skills_score,
        'experience': experience_score,
        'education': education_score,
    }


def main():
    """Main entry point for the program."""
    resume_path = "65456466.pdf"
    job_desc_path = "job-desc.txt"

    # Calculate match results
    results = calculate_resume_match(resume_path, job_desc_path)

    # Display results
    print("\nDetailed Results:")
    for key, value in results.items():
        print(f"{key.capitalize()}: {value}")


if __name__ == "__main__":
    main()
