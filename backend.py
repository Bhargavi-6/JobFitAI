from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import fitz  # PyMuPDF for PDF extraction
import docx
import io
import os
import logging
import numpy as np
import hashlib
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# LangChain LLM Setup
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.3
)

# -------------------------------
# FastAPI App Initialization
# -------------------------------
app = FastAPI(title="Resume Matcher API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Constants
# -------------------------------
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# -------------------------------
# JD Cache for Skills
# -------------------------------
jd_cache = {}

# -------------------------------
# Helper Functions
# -------------------------------
def validate_file_size(file: UploadFile):
    """Validate uploaded file size."""
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File {file.filename} exceeds 5MB.")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF files."""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text("text") for page in pdf)
    pdf.close()
    return text.strip()

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX files."""
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs).strip()

def extract_text(file: UploadFile) -> str:
    """Determine file type and extract text."""
    validate_file_size(file)
    content = file.file.read()
    file.file.seek(0)
    if file.filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif file.filename.lower().endswith(".docx"):
        return extract_text_from_docx(content)
    elif file.filename.lower().endswith(".txt"):
        return content.decode("utf-8")
    else:
        raise ValueError("Unsupported file format. Only PDF, DOCX, and TXT are supported.")

def convert_numpy_types(obj):
    """Converts numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    return obj

# -------------------------------
# LLM Call #1 - Extract Skills from JD
# -------------------------------
def extract_skills_from_jd(jd_text: str) -> list:
    """
    Extract key skills from a JD using LLM.
    Cache is used to avoid duplicate calls.
    """
    jd_hash = hashlib.md5(jd_text.encode("utf-8")).hexdigest()

    # Return from cache if present
    if jd_hash in jd_cache:
        logger.info("Returning cached JD skills.")
        return jd_cache[jd_hash]

    prompt_template = PromptTemplate(
        input_variables=["jd_text"],
        template=(
            "You are an expert HR assistant. Extract only the key skills from the following job description. "
            "Provide the output strictly as a comma-separated list of skills.\n\n"
            "Job Description:\n{jd_text}"
        ),
    )

    skill_chain = LLMChain(llm=llm, prompt=prompt_template)
    try:
        response = skill_chain.run(jd_text=jd_text)
        skills = [skill.strip() for skill in response.split(",") if skill.strip()]
        jd_cache[jd_hash] = skills
        return skills
    except Exception as e:
        logger.error(f"Skill extraction failed: {e}")
        return []

# -------------------------------
# LLM Call #2 - Resume Summarization
# -------------------------------
def generate_resume_summary(resume_text: str) -> str:
    """
    Generate a concise resume summary using LLM.
    """
    prompt_template = PromptTemplate(
        input_variables=["resume_text"],
        template=(
            "You are an expert HR assistant. Summarize the following resume in 3-5 concise bullet points, "
            "highlighting the candidate's key skills, experience, and achievements:\n\n"
            "{resume_text}"
        ),
    )
    summary_chain = LLMChain(llm=llm, prompt=prompt_template)
    try:
        response = summary_chain.run(resume_text=resume_text)
        return response.strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return " ".join(resume_text.split()[:200])

# -------------------------------
# Similarity Calculation
# -------------------------------
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    sentence_model = None
    logger.warning(f"Sentence Transformer failed: {e}. Falling back to TF-IDF only.")

def calculate_enhanced_similarity(jd_text, resume_text, jd_requirements) -> Dict[str, float]:
    """Calculate semantic, TF-IDF, and keyword similarity with weighted scoring."""
    scores = {}

    # Semantic similarity
    if sentence_model:
        jd_emb = sentence_model.encode([jd_text])
        res_emb = sentence_model.encode([resume_text])
        scores['semantic'] = float(cosine_similarity(jd_emb, res_emb)[0][0] * 100)
    else:
        scores['semantic'] = 0.0

    # TF-IDF similarity
    tfidf_matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([jd_text, resume_text])
    scores['tfidf'] = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100)

    # Keyword matching
    if 'skills' in jd_requirements and isinstance(jd_requirements['skills'], list):
        total = len(jd_requirements['skills'])
        found = sum(1 for s in jd_requirements['skills'] if s.lower() in resume_text.lower())
        scores['keyword'] = round((found / total) * 100 if total > 0 else 0.0, 2)
    else:
        scores['keyword'] = 0.0

    # Weighted overall score
    weights = {'semantic': 0.4, 'tfidf': 0.4, 'keyword': 0.2}
    scores['overall'] = round(sum(scores[k] * weights[k] for k in weights), 2)
    return scores

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/analyze")
async def analyze(jd_file: UploadFile = File(...), resume_files: List[UploadFile] = File(...)):
    """
    Analyze JD and Resumes:
    - Extract JD skills (LLM call #1 with caching)
    - Generate resume summaries (LLM call #2)
    - Calculate match scores
    """
    try:
        # Extract JD text
        jd_text = extract_text(jd_file)

        # Extract JD skills
        skills = extract_skills_from_jd(jd_text)
        jd_requirements = {"skills": skills}

        logger.info(f"Extracted Skills: {skills}")

        all_results, qualified_results = [], []
        for resume in resume_files:
            resume_text = extract_text(resume)

            # Calculate similarity
            scores = calculate_enhanced_similarity(jd_text, resume_text, jd_requirements)

            # Generate summary
            summary = generate_resume_summary(resume_text)

            result = {
                "filename": resume.filename,
                "match_score": scores['overall'],
                "detailed_scores": scores,
                "summary": summary,
                "is_qualified": bool(scores['overall'] >= 80.0)
            }

            all_results.append(result)
            if result['is_qualified']:
                qualified_results.append(result)

        # Sort by match score
        all_results.sort(key=lambda x: x['match_score'], reverse=True)
        qualified_results.sort(key=lambda x: x['match_score'], reverse=True)

        return convert_numpy_types({
            "jd_requirements": jd_requirements,
            "all_matches": all_results,
            "qualified_matches": qualified_results,
            "total_resumes": len(all_results),
            "qualified_count": len(qualified_results)
        })

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return {"error": str(e)}

# -------------------------------
# Health Check Endpoint
# -------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0"}

# -------------------------------
# Run the App
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

