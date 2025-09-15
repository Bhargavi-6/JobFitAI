# JobFitAI
AI-powered resume-job fit analysis.


# 📄 AI Resume Matcher  

AI-powered platform to **analyze job descriptions and resumes** to identify the most qualified candidates.  
This project leverages **LLMs (OpenAI GPT-4o-mini)**, **semantic similarity**, and **TF-IDF-based matching** to score resumes against a job description and generate concise summaries of candidate profiles.

---

## 🚀 Project Overview  

Recruiters often need to manually sift through hundreds of resumes to find the best fit for a role.  
This application **automates resume screening** by:  

- Extracting **key skills** from a Job Description (JD).  
- Generating **summaries of resumes** to highlight strengths.  
- Calculating a **match score** between the JD and each resume using multiple techniques:  
  - **Semantic Similarity** via Sentence Transformers.  
  - **TF-IDF Similarity** for text pattern matching.  
  - **Keyword Matching** based on JD skills.  
- Categorizing candidates as:
  - ✅ **Qualified**
  - ⚠️ **Review**
  - ❌ **Not Qualified**

### Tech Stack
| **Component**        | **Technology Used**           |
|-----------------------|-------------------------------|
| **Frontend**         | Streamlit                     |
| **Backend**          | FastAPI                       |
| **LLM**              | OpenAI GPT-4o-mini via LangChain |
| **Embeddings**       | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Matching Models**  | TF-IDF, Cosine Similarity     |
| **File Handling**    | PyMuPDF (PDF), python-docx (DOCX) |
| **Deployment Ready** | Uvicorn (API) + Streamlit UI |

---

## 🗂 Project Structure
```
AI-Resume-Matcher/
│
├── app.py              # Streamlit frontend
├── backend.py          # FastAPI backend
├── requirements.txt    # Python dependencies
├── .env                # API key storage (add OPENAI_API_KEY here)
└── README.md           # Project documentation
```

---

## ⚙️ Setup Instructions  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Bhargavi-6/JobFitAI.git
cd JobFitAI
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🔑 Configure API Key  
Create a `.env` file in the root directory and add your **OpenAI API Key**:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ Running the Application  

### **Step 1: Start Backend (FastAPI)**
Runs on **http://127.0.0.1:8000**  
```bash
uvicorn backend:app --reload --port 8000
```

### **Step 2: Start Frontend (Streamlit UI)**
Runs on **http://localhost:8501**  
```bash
streamlit run app.py
```

---

## 🖥 Usage Instructions

1. **Upload the Job Description (JD)**  
   - Supports `PDF`, `DOCX`, and `TXT` formats (max 5 MB).  
2. **Upload one or more resumes** in supported formats.  
3. **Adjust Analysis Settings:**
   - Set **Minimum Match Score Threshold** (default: 80%).  
   - Toggle **Show All Candidates** to view unqualified resumes.  
4. Click **Start Enhanced Analysis**.  
5. **Results Displayed:**
   - Overall match score per resume.  
   - Category: Qualified / Review / Not Qualified.  
   - AI-generated **resume summary**.  

---

## 📊 Methods & Workflow

### **Step 1: Text Extraction**
- **PDFs** → Extracted using PyMuPDF (`fitz`).  
- **DOCX** → Extracted using `python-docx`.  
- **TXT** → Read directly.

### **Step 2: LLM Processing**
- **Call #1:** Extract **key skills** from the Job Description using GPT-4o-mini.  
  - Example prompt:
    > *"Extract only the key skills from the job description as a comma-separated list."*
- **Call #2:** Generate a **concise resume summary**.  
  - Example prompt:
    > *"Summarize the resume in 3-5 bullet points highlighting skills, experience, and achievements."*

### **Step 3: Matching Algorithm**
The match score is a **weighted combination** of:
| **Technique**      | **Weight** | **Purpose** |
|---------------------|------------|--------------|
| **Semantic Similarity** (Sentence Transformers) | 40% | Contextual understanding between JD and resume. |
| **TF-IDF Similarity** | 40% | Surface-level keyword relevance. |
| **Keyword Matching** (from JD skills) | 20% | Explicit skill presence check. |

**Formula:**  
```
Overall Score = (0.4 * Semantic) + (0.4 * TF-IDF) + (0.2 * Keyword)
```

---

## 🧪 Example Output

| **Candidate** | **Score** | **Status** |
|---------------|-----------|------------|
| John_Doe.pdf  | 92%       | ✅ Qualified |
| Jane_Smith.pdf| 78%       | ⚠️ Review |
| Alex_Khan.pdf | 60%       | ❌ Not Qualified |

Each result also includes:
- Summary of the resume.
- Detailed breakdown of semantic, TF-IDF, and keyword scores.

### App Screenshot

![App Screenshot](https://github.com/Bhargavi-6/JobFitAI/blob/main/images/image-1.png?raw=true)

!![App Screenshot](https://github.com/Bhargavi-6/JobFitAI/blob/main/images/image-2.png?raw=true)

![App Screenshot](https://github.com/Bhargavi-6/JobFitAI/blob/main/images/image-3.png?raw=true)

![App Screenshot](https://github.com/Bhargavi-6/JobFitAI/blob/main/images/image-4.png?raw=true)

![App Screenshot](https://github.com/Bhargavi-6/JobFitAI/blob/main/images/image-5.png?raw=true)

---

## 🌐 API Endpoints  

| **Endpoint**      | **Method** | **Description** |
|--------------------|------------|-----------------|
| `/analyze`         | POST       | Analyze JD + resumes and return match results |
| `/health`          | GET        | Health check endpoint |

---

## 📦 Dependencies
See [`requirements.txt`](requirements.txt) for full details.

Key libraries:
- **FastAPI** - Backend API.
- **Streamlit** - Frontend interface.
- **LangChain + OpenAI** - LLM integration.
- **Sentence Transformers** - Semantic similarity.
- **PyMuPDF & python-docx** - File parsing.
- **scikit-learn** - TF-IDF & cosine similarity.

---



