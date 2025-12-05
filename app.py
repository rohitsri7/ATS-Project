import streamlit as st
import google.generativeai as genai
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="ATS Resume Pro", layout="wide")

# 1. Load NLP Model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Spacy model not found. Please run: python -m spacy download en_core_web_sm")

# 2. Configure Gemini API
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("API Key missing! Please set up .streamlit/secrets.toml")

# --- FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text from uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text):
    """Cleans text for mathematical scoring."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove special chars
    return text

def calculate_similarity(resume_text, jd_text):
    """Calculates percentage match based on vector cosine similarity."""
    documents = [clean_text(resume_text), clean_text(jd_text)]
    vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix = vectorizer.fit_transform(documents)
    score_matrix = cosine_similarity(sparse_matrix, sparse_matrix)
    return round(score_matrix[0][1] * 100, 2)

def get_gemini_analysis(resume_text, jd_text):
    """Sends text to Gemini AI for intelligent suggestions."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Act as a highly experienced Technical Recruiter and ATS Specialist.
    Review the following Resume against the Job Description (JD).
    
    RESUME TEXT:
    {resume_text[:3000]}
    
    JOB DESCRIPTION:
    {jd_text[:3000]}
    
    Provide a professional analysis in this format:
    1. **Missing Keywords:** List the top 5 hard skills/keywords missing from the resume.
    2. **Profile Summary Rewrite:** Write a new 3-sentence professional summary tailored to this JD.
    3. **Bullet Point Improvements:** Pick 2 weak bullet points from the resume and rewrite them to be result-oriented (using numbers/metrics) and keyword-rich.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

# --- UI LAYOUT ---

st.title("🚀 AI Resume Optimizer & ATS Checker")
st.markdown("Optimize your resume for Applicant Tracking Systems (ATS) instantly.")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Your Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

with col2:
    st.subheader("2. Job Description")
    jd_input = st.text_area("Paste the Job Description here...", height=200)

# Analysis Section
if st.button("Analyze My Resume", type="primary"):
    if uploaded_file is not None and jd_input:
        
        with st.spinner("Parsing resume and calculating scores..."):
            # Extract Text
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # Calculate Score
            score = calculate_similarity(resume_text, jd_input)
            
        # Display Score with Gauge
        st.divider()
        st.header(f"ATS Match Score: {score}%")
        st.progress(score / 100)
        
        if score >= 80:
            st.success("✅ Great match! Your resume is well-aligned.")
        elif score >= 50:
            st.warning("⚠️ Average match. Focus on the suggestions below.")
        else:
            st.error("❌ Low match. Needs significant optimization.")

        # AI Analysis
        st.divider()
        st.subheader("🤖 AI Detailed Analysis")
        with st.spinner("Generating AI suggestions (this takes 5 seconds)..."):
            ai_response = get_gemini_analysis(resume_text, jd_input)
            st.markdown(ai_response)
            
    else:
        st.warning("Please upload a resume AND paste a job description first.")
