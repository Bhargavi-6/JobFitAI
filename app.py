import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000/analyze"

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    layout="wide",
    page_icon="📄",
)

# ---------------------------
# Custom CSS for Enhanced UI
# ---------------------------
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    div.block-container { padding: 0.5rem 1rem; }
    .qualified-badge { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 8px; }
    .not-qualified-badge { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 8px; }
    .score-excellent { color: #28a745; font-weight: bold; }
    .score-good { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
    .metric-card { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0; }
    button[kind="primary"] { background-color: #007bff !important; color: white !important; border-radius: 6px; }
    .stProgress > div > div > div { background-color: #007bff !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Main Header
# ---------------------------
st.markdown("""
<div style="padding-top:20px; padding-bottom:25px; text-align: center;">
    <h1 style='color: #2c3e50; margin-bottom: 8px;'>📄 AI Resume Matcher</h1>
    <p style='color: #6c757d; font-size: 16px; margin-bottom: 4px;'>
        Upload Job Description and Resumes to find qualified candidates
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Layout: Two Columns
# ---------------------------
left_col, right_col = st.columns([1, 2], gap="large")

# ---------------------------
# LEFT PANEL - Upload Section
# ---------------------------
with left_col:
    st.markdown("### 📁 Upload Files")
    
    jd_file = st.file_uploader(
        "Upload JD (PDF/DOCX/TXT, max 5MB)",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
        help="Upload the job description to match against"
    )
    
    resumes = st.file_uploader(
        "Upload Resumes (PDF/DOCX/TXT, max 5MB each)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more resume files"
    )
    
    if resumes:
        st.markdown(f"📄 **{len(resumes)} resume(s) selected**")
        for resume in resumes:
            st.markdown(f"• {resume.name}")
    
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    
    # Analysis Settings
    with st.expander("⚙️ Analysis Settings"):
        min_score = st.slider("Minimum qualification score", 70, 95, 80, 5)
        show_all = st.checkbox("Show all candidates (not just qualified)", value=False)
    
    analyze_btn = st.button("Start Enhanced Analysis", type="primary", use_container_width=True)

# ---------------------------
# RIGHT PANEL - Results Section  
# ---------------------------
with right_col:
    st.markdown("### 📊 Analysis Results")
    
    if analyze_btn:
        if jd_file is None or not resumes:
            st.error("Please upload a Job Description and at least one Resume.")
        else:
            with st.spinner("Performing enhanced AI analysis..."):
                try:
                    # Send min_score as part of request data
                    data_payload = {"min_score": min_score}

                    files = {"jd_file": (jd_file.name, jd_file, jd_file.type)}
                    resume_files = [("resume_files", (resume.name, resume, resume.type)) for resume in resumes]
                    merged_files = list(files.items()) + resume_files
                    
                    response = requests.post(API_URL, files=merged_files, data=data_payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if "error" in data:
                            st.error(f"Error: {data['error']}")
                        else:
                            qualified_count = data.get("qualified_count", 0)
                            total_count = data.get("total_resumes", 0)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Resumes", total_count)
                            with col2:
                                st.metric(f"Qualified ({min_score}%+)", qualified_count, delta=f"{qualified_count}/{total_count}")
                            with col3:
                                qualification_rate = (qualified_count/total_count*100) if total_count > 0 else 0
                                st.metric("Qualification Rate", f"{qualification_rate:.1f}%")
                            
                            if show_all:
                                matches_to_show = data.get("all_matches", [])
                                st.markdown("### 📄 All Candidates")
                            else:
                                matches_to_show = data.get("qualified_matches", [])
                                st.markdown(f"### ✅ Qualified Candidates ({len(matches_to_show)})")
                            
                            if matches_to_show:
                                for match in matches_to_show:
                                    match_score = match.get('match_score', 0)
                                    
                                    # Dynamically using min_score for qualification
                                    if match_score >= min_score:
                                        score_class = "score-excellent"
                                        badge_class = "qualified-badge"
                                        badge_text = "QUALIFIED"
                                    elif match_score >= (min_score - 20):
                                        score_class = "score-good"
                                        badge_class = "not-qualified-badge"
                                        badge_text = "REVIEW"
                                    else:
                                        score_class = "score-poor"
                                        badge_class = "not-qualified-badge"
                                        badge_text = "NOT QUALIFIED"
                                    
                                    st.markdown(
                                        f"""
                                        <div class="metric-card">
                                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                                <h4 style="margin: 0; color: #2c3e50;">📂 {match['filename']}</h4>
                                                <span class="{badge_class}">{badge_text}</span>
                                            </div>
                                            <div style="margin-bottom: 8px;">
                                                <span class="{score_class}" style="font-size: 18px;">
                                                    Overall Score: {match_score}%
                                                </span>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
                                    st.progress(match_score / 100)
                                    
                                    with st.expander(f"📝 Summary - {match['filename']}"):
                                        st.markdown("**Candidate Summary:**")
                                        st.write(match.get('summary', 'Not available'))
                                    
                                    st.markdown("---")
                                
                                if qualified_count > 0:
                                    st.success(f"🎉 Found {qualified_count} candidate(s) meeting the {min_score}% threshold!")
                                else:
                                    st.warning(f"⚠️ No candidates met the {min_score}% qualification threshold.")
                            else:
                                st.warning("No candidates found.")
                    
                    else:
                        st.error(f"Backend error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
    else:
        st.info("👈 Upload files on the left and click **Start Enhanced Analysis** to begin.")
