import PyPDF2
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def read_job_description(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def extract_info(text, question):
    result = qa_model(question=question, context=text, max_answer_len=100)
    return result['answer']

def semantic_similarity(text1, text2):
    embeddings1 = similarity_model.encode(text1, convert_to_tensor=True)
    embeddings2 = similarity_model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return similarity

def extract_keywords(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 1]

def keyword_match_score(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

def calculate_resume_match(resume_file, job_desc_file):
    job_description = read_job_description(job_desc_file)
    resume_text = extract_text_from_pdf(resume_file)

    overall_score = round(keyword_match_score(job_description, resume_text))

    job_title_desc = extract_info(job_description, "What is the job title and main responsibilities?")
    job_title_resume = extract_info(resume_text, "What is the applicant's current job title and main responsibilities?")
    
    skills_desc = extract_info(job_description, "What are the key skills, technologies, and qualifications required for this job?")
    skills_resume = extract_info(resume_text, "What are the applicant's key skills, technologies, and qualifications?")
    
    experience_desc = extract_info(job_description, "What kind of work experience and achievements are required for this job?")
    experience_resume = extract_info(resume_text, "What is the applicant's work experience and key achievements?")
    
    education_desc = extract_info(job_description, "What educational background and certifications are required for this job?")
    education_resume = extract_info(resume_text, "What is the applicant's educational background and certifications?")

    job_title_score = round(semantic_similarity(job_title_desc, job_title_resume) * 100)
    skills_score = round(keyword_match_score(skills_desc, skills_resume))
    experience_score = round(semantic_similarity(experience_desc, experience_resume) * 100)
    education_score = round(semantic_similarity(education_desc, education_resume) * 100)

    match_score = overall_score

    print(f"Overall Match Score: {match_score}/100")

    return {
        'match_score': match_score,
        'job_title': job_title_score,
        'skills': skills_score,
        'experience': experience_score,
        'education': education_score,
    }

if __name__ == "__main__":
    results = calculate_resume_match("resume.pdf", "job-desc.txt")
