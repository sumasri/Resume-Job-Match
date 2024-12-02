import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
# Make sure to install the required packages:
# pip install transformers torch
from transformers import pipeline

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def read_job_description(file_path):
    """Read job description from a text file."""
    with open(file_path, 'r') as file:
        return file.read()

def get_match_score(resume_text, job_description):
    """Use a Hugging Face model to get a match score between resume and job description."""
    # Load a pre-trained model for text classification
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Combine resume and job description for analysis
    combined_text = f"Resume: {resume_text}\n\nJob Description: {job_description}"
    
    # Get the sentiment score (as a proxy for match score)
    result = classifier(combined_text)[0]
    
    # Convert the sentiment score to a match score out of 100
    match_score = round(result['score'] * 100)
    
    # Prepare the explanation
    explanation = f"Based on the analysis, the resume {'matches well' if result['label'] == 'POSITIVE' else 'does not match well'} with the job description."
    
    return f"Score (out of 100): {match_score}\nExplanation: {explanation}"

def main():
    resume_path = "resume.pdf"
    job_desc_path = "job-desc.txt"

    resume_text = extract_text_from_pdf(resume_path)
    job_description = read_job_description(job_desc_path)

    match_result = get_match_score(resume_text, job_description)
    print(match_result)

if __name__ == "__main__":
    main()
