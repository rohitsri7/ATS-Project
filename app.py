import streamlit as st
import google.generativeai as genai
import fitz  # This is PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="Visual ATS Resume Optimizer", layout="wide")

# 1. Load NLP Model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Spacy model not found. Check requirements.txt")

# 2. Configure Gemini API
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("API Key missing. Please set it in .streamlit/secrets.toml")

# --- CORE FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text using PyMuPDF (faster & better than PyPDF2)."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_highlighted_pdf(uploaded_file, phrases_to_highlight):
    """
    1. Opens the PDF.
    2. Searches for the exact phrases.
    3. Draws a red box around them.
    4. Returns an image of the first page with highlights.
    """
    # Reset file pointer to beginning
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    first_page = doc[0]  # We only highlight the first page for now

    for phrase in phrases_to_highlight:
        # Search for the text on the page
        text_instances = first_page.search_for(phrase)
        
        # Draw a red rectangle (stroke) around every instance found
        for inst in text_instances:
            first_page.draw_rect(inst, color=(1, 0, 0), width=2) # Red box

    # Convert page to image (Pixmap) so Streamlit can show it
    pix = first_page.get_pixmap()
    return pix.tobytes()

def calculate_similarity(resume_text, jd_text):
    # Simple cleaning
    text1 = re.sub(r'[^a-z\s]', '', resume_text.lower())
    text2 = re.sub(r'[^a-z\s]', '', jd_text.lower())
    
    documents = [text1, text2]
    vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(sparse_matrix, sparse_matrix)[0][1]
    return round(score * 100, 2)

def get_gemini_analysis(resume_text, jd_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # We ask Gemini to return JSON so we can programmatically use the text
    prompt = f"""
    Act as an ATS Expert. 
    RESUME: {resume_text[:3000]}
    JOB DESCRIPTION: {jd_text[:3000]}
    
    Identify 3 specific sentences or bullet points in the resume that are WEAK or generic.
    Return a strictly valid JSON object with this structure:
    {{
        "weak_phrases": [
            "exact substring from resume 1",
            "exact substring from resume 2",
            "exact substring from resume 3"
        ],
        "advice": [
            "Advice for phrase 1",
            "Advice for phrase 2",
            "Advice for phrase 3"
        ]
    }}
    Important: The "weak_phrases" must match the resume text EXACTLY (character for character) so I can search and highlight them.
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        return {"weak_phrases": [], "advice": [f"Error: {e}"]}

# --- APP UI ---

st.title("🖌️ Visual ATS Optimizer")
st.markdown("This tool **draws red boxes** on your resume where changes are needed.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_input = st.text_area("Paste Job Description", height=200)

if st.button("Analyze & Highlight", type="primary"):
    if uploaded_file and jd_input:
        
        with st.spinner("Reading PDF and calculating score..."):
            # 1. Extract Text
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # 2. Score
            score = calculate_similarity(resume_text, jd_input)
            
        st.header(f"Match Score: {score}%")
        st.progress(score/100)

        with st.spinner("AI is identifying weak spots and drawing boxes..."):
            # 3. Get Weak Phrases from Gemini
            analysis = get_gemini_analysis(resume_text, jd_input)
            weak_phrases = analysis.get("weak_phrases", [])
            advice_list = analysis.get("advice", [])

            # 4. Draw boxes on the PDF
            if weak_phrases:
                highlighted_image_bytes = create_highlighted_pdf(uploaded_file, weak_phrases)
                
                # Show Side-by-Side: Image vs Advice
                img_col, text_col = st.columns(2)
                
                with img_col:
                    st.subheader("Your Marked Resume")
                    st.image(highlighted_image_bytes, caption="Red boxes indicate weak areas.", use_container_width=True)
                
                with text_col:
                    st.subheader("Suggested Changes")
                    for i, (phrase, advice) in enumerate(zip(weak_phrases, advice_list)):
                        st.warning(f"**Issue #{i+1}:** \"{phrase}...\"")
                        st.info(f"💡 **Fix:** {advice}")
            else:
                st.warning("AI couldn't find specific sentences to highlight, but here is general advice.")
                st.write(analysis)
                
    else:
        st.error("Please upload a file and paste a JD.")
