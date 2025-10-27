import os
import re
import pickle
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import requests

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RESUME_FOLDER = os.getenv("RESUME_FOLDER", "resumes")
EMBEDDING_CACHE = os.getenv("EMBEDDING_CACHE", "resume_embeddings.pkl")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

Path(RESUME_FOLDER).mkdir(parents=True, exist_ok=True)

# Enhanced regex patterns
EMAIL_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    r'\b[A-Za-z0-9]+[\._]?[A-Za-z0-9]+[@]\w+[.]\w{2,3}\b',
    r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    r'Email\s*:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    r'E-mail\s*:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
]

PHONE_PATTERNS = [
    r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
    r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
    r'\b\d{10}\b',
    r'(?:Phone|Mobile|Tel|Contact)[\s:]*([+\d\s\-\(\)]+)',
]

# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
MODEL_DIM = model.get_sentence_embedding_dimension()

# -----------------------------
# Enhanced Email/Phone Extraction
# -----------------------------
def extract_emails_advanced(text: str) -> List[str]:
    """Extract emails using multiple patterns and validation"""
    emails = set()
    
    # Try all patterns
    for pattern in EMAIL_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            email = match.strip().lower()
            # Validate email
            if '@' in email and '.' in email.split('@')[-1]:
                # Remove common false positives
                if not any(x in email for x in ['example.com', 'domain.com', 'email.com', 'test.com']):
                    emails.add(email)
    
    return sorted(list(emails), key=lambda x: (
        '@gmail.com' not in x,
        '@outlook.com' not in x,
        '@yahoo.com' not in x,
        len(x)
    ))

def extract_phones_advanced(text: str) -> List[str]:
    """Extract phone numbers using multiple patterns"""
    phones = set()
    
    for pattern in PHONE_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            phone = re.sub(r'[^\d+]', '', match.strip())
            if 10 <= len(phone) <= 15:
                phones.add(match.strip())
    
    return list(phones)

def extract_name_advanced(text: str) -> str:
    """Extract candidate name intelligently"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Skip common headers
    skip_words = ['resume', 'cv', 'curriculum vitae', 'profile', 'contact', 'email', 'phone']
    
    for line in lines[:10]:
        line_lower = line.lower()
        # Skip if contains email or phone
        if '@' in line or re.search(r'\d{10}', line):
            continue
        # Skip common headers
        if any(word in line_lower for word in skip_words):
            continue
        # Check if looks like a name (2-4 words, mostly letters)
        words = line.split()
        if 2 <= len(words) <= 4:
            if all(len(w) > 1 and w[0].isupper() for w in words):
                return line[:50]
    
    return lines[0][:50] if lines else "Unknown"

# -----------------------------
# Ollama Integration with Better Error Handling
# -----------------------------
def call_ollama(prompt: str, system_prompt: str = None, temperature: float = 0.3, max_retries: int = 2) -> str:
    """Enhanced Ollama API call with retries"""
    for attempt in range(max_retries):
        try:
            url = f"{OLLAMA_API_URL}/api/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "num_predict": 1000
            }
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                if attempt < max_retries - 1:
                    continue
                return f"Error: Status {response.status_code}"
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return f"Error: {str(e)}"
    return ""

# -----------------------------
# Enhanced Contact Extraction with LLM
# -----------------------------
def extract_contact_info_hybrid(resume_text: str, filename: str = None) -> Dict[str, Optional[str]]:
    """
    Hybrid approach: Regex first, then LLM fallback
    Returns: {name, email, phone, linkedin}
    """
    # Cache key
    if filename and 'contact_cache' not in st.session_state:
        st.session_state['contact_cache'] = {}
    
    if filename and filename in st.session_state['contact_cache']:
        return st.session_state['contact_cache'][filename]
    
    # Step 1: Regex extraction
    emails = extract_emails_advanced(resume_text)
    phones = extract_phones_advanced(resume_text)
    name = extract_name_advanced(resume_text)
    
    # Step 2: LLM fallback for missing info
    if not emails:
        llm_prompt = f"""Extract ONLY the email address from this resume. Return ONLY the email, nothing else.

Resume text:
{resume_text[:2500]}

If no email found, respond with: NONE

Email address:"""
        
        response = call_ollama(llm_prompt, temperature=0.1).strip()
        
        # Extract email from response
        llm_emails = extract_emails_advanced(response)
        if llm_emails:
            emails = llm_emails
    
    if not phones:
        llm_prompt = f"""Extract ONLY the phone number from this resume. Return ONLY the phone number, nothing else.

Resume text:
{resume_text[:2500]}

If no phone found, respond with: NONE

Phone number:"""
        
        response = call_ollama(llm_prompt, temperature=0.1).strip()
        llm_phones = extract_phones_advanced(response)
        if llm_phones:
            phones = llm_phones
    
    # Extract LinkedIn
    linkedin = None
    linkedin_match = re.search(r'linkedin\.com/in/([a-zA-Z0-9\-]+)', resume_text, re.IGNORECASE)
    if linkedin_match:
        linkedin = f"linkedin.com/in/{linkedin_match.group(1)}"
    
    result = {
        "name": name,
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "linkedin": linkedin
    }
    
    # Cache result
    if filename:
        st.session_state['contact_cache'][filename] = result
    
    return result

# -----------------------------
# Enhanced ATS Scoring System
# -----------------------------
def calculate_ats_score_enhanced(resume_text: str, job_description: str, filename: str = None) -> dict:
    """
    Advanced ATS scoring with improved prompt engineering
    """
    # Check cache
    if filename:
        cache_key = f"{filename}_{hash(job_description)}"
        if 'ats_cache' not in st.session_state:
            st.session_state['ats_cache'] = {}
        
        if cache_key in st.session_state['ats_cache']:
            return st.session_state['ats_cache'][cache_key]
    
    system_prompt = """You are an expert ATS (Applicant Tracking System) analyzer. 
Your job is to evaluate resumes against job descriptions with high precision.
Focus on: keyword matching, skills alignment, experience relevance, education fit.
Be objective, thorough, and strict in your evaluation."""
    
    # Extract key requirements from JD
    jd_keywords = extract_key_terms(job_description)
    
    prompt = f"""Analyze this resume against the job requirements and provide detailed ATS scoring.

JOB DESCRIPTION:
{job_description[:2000]}

KEY REQUIREMENTS: {', '.join(jd_keywords[:15])}

RESUME:
{resume_text[:4000]}

Provide evaluation in VALID JSON format ONLY. No extra text.

{{
    "ats_score": 85,
    "keyword_match_score": 80,
    "skills_match_score": 85,
    "experience_score": 90,
    "education_score": 80,
    "overall_grade": "A",
    "matched_keywords": ["python", "machine learning", "sql"],
    "missing_keywords": ["kubernetes", "docker"],
    "matched_skills": ["data analysis", "statistics"],
    "missing_skills": ["cloud computing"],
    "years_of_experience": 5,
    "education_level": "Masters",
    "key_strengths": ["Strong technical skills", "Relevant experience"],
    "red_flags": [],
    "hire_recommendation": "Strong Hire",
    "confidence_level": "High",
    "detailed_notes": "Excellent match for role requirements"
}}

SCORING RULES:
- ats_score: Overall match (0-100)
- Score 85+ = Strong Hire, 70-84 = Hire, 60-69 = Maybe, <60 = Reject
- Be strict: Only give 85+ for excellent matches
- keyword_match_score: % of required keywords found
- skills_match_score: Skills alignment
- experience_score: Years and relevance
- education_score: Degree and field match

Respond with ONLY the JSON object:"""
    
    response = call_ollama(prompt, system_prompt, temperature=0.2)
    
    # Parse JSON with better error handling
    try:
        # Extract JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            raise ValueError("No JSON found in response")
        
        json_str = response[json_start:json_end]
        
        # Clean JSON
        json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        json_str = ' '.join(json_str.split())
        
        result = json.loads(json_str)
        
        # Validate required fields
        required = ['ats_score', 'keyword_match_score', 'skills_match_score', 
                    'experience_score', 'education_score', 'overall_grade']
        
        if all(field in result for field in required):
            # Validate scores are integers
            for field in ['ats_score', 'keyword_match_score', 'skills_match_score', 
                          'experience_score', 'education_score']:
                if not isinstance(result[field], (int, float)):
                    result[field] = 65
                result[field] = max(0, min(100, int(result[field])))
            
            # Cache result
            if filename:
                st.session_state['ats_cache'][cache_key] = result
            
            return result
        
    except Exception as e:
        st.warning(f"⚠️ Parsing issue for {filename}: {str(e)[:100]}")
    
    # Enhanced fallback with keyword matching
    keywords_found = sum(1 for kw in jd_keywords if kw.lower() in resume_text.lower())
    keyword_score = min(100, int((keywords_found / max(len(jd_keywords), 1)) * 100))
    
    result = {
        "ats_score": max(50, keyword_score - 10),
        "keyword_match_score": keyword_score,
        "skills_match_score": 60,
        "experience_score": 65,
        "education_score": 60,
        "overall_grade": "C+",
        "matched_keywords": jd_keywords[:min(keywords_found, 5)],
        "missing_keywords": jd_keywords[keywords_found:keywords_found+5],
        "matched_skills": ["Analysis pending"],
        "missing_skills": ["Full analysis needed"],
        "years_of_experience": 0,
        "education_level": "Unknown",
        "key_strengths": ["Resume parsed successfully"],
        "red_flags": ["LLM analysis unavailable"],
        "hire_recommendation": "Review Required",
        "confidence_level": "Medium",
        "detailed_notes": "Fallback scoring used. Manual review recommended."
    }
    
    if filename:
        st.session_state['ats_cache'][cache_key] = result
    
    return result

def extract_key_terms(text: str, max_terms: int = 20) -> List[str]:
    """Extract key technical terms and skills from job description"""
    # Common tech skills and keywords
    tech_keywords = ['python', 'java', 'javascript', 'sql', 'aws', 'azure', 'docker', 
                     'kubernetes', 'react', 'node', 'machine learning', 'ai', 'data',
                     'cloud', 'devops', 'agile', 'api', 'database', 'frontend', 'backend']
    
    text_lower = text.lower()
    found = []
    
    for keyword in tech_keywords:
        if keyword in text_lower:
            found.append(keyword)
    
    # Extract multi-word terms
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    word_freq = {}
    for word in words:
        if word not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'have']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top frequent words
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_terms]
    found.extend([w[0] for w in top_words if w[0] not in found])
    
    return found[:max_terms]

# -----------------------------
# PDF and Embedding Functions
# -----------------------------
def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for p in reader.pages:
                page_text = p.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        st.error(f"❌ Error reading {path}: {e}")
    return text

def load_embeddings_cache():
    if os.path.exists(EMBEDDING_CACHE):
        try:
            with open(EMBEDDING_CACHE, "rb") as f:
                return pickle.load(f)
        except:
            return {}
    return {}

def save_embeddings_cache(cache):
    with open(EMBEDDING_CACHE, "wb") as f:
        pickle.dump(cache, f)

def ensure_embeddings_for_resumes():
    cache = load_embeddings_cache()
    
    # Validate cache
    regenerate = False
    for k, v in cache.items():
        e = v.get("embedding")
        if not isinstance(e, torch.Tensor) or e.shape[0] != MODEL_DIM:
            regenerate = True
            break
    
    if regenerate:
        cache = {}
    
    changed = False
    for filename in sorted(os.listdir(RESUME_FOLDER)):
        if not filename.lower().endswith(".pdf"):
            continue
        if filename in cache:
            continue
        
        path = os.path.join(RESUME_FOLDER, filename)
        text = extract_text_from_pdf(path)
        
        if text.strip():
            emb = model.encode(text, convert_to_tensor=True).cpu()
            cache[filename] = {"embedding": emb, "text": text}
            changed = True
    
    if changed:
        save_embeddings_cache(cache)
    
    return cache

def advanced_shortlist(job_desc: str, top_k: int = 20):
    """Shortlist candidates based on semantic similarity"""
    cache = ensure_embeddings_for_resumes()
    jd_emb = model.encode(job_desc, convert_to_tensor=True).cpu()
    
    rows = []
    for fname, v in cache.items():
        sim = float(util.cos_sim(jd_emb, v["embedding"]).item())
        rows.append({
            "filename": fname, 
            "similarity": sim, 
            "text": v["text"]
        })
    
    df = pd.DataFrame(rows).sort_values("similarity", ascending=False).head(top_k)
    return df

# -----------------------------
# Email Sending
# -----------------------------
def generate_jitsi_link(job_title: str):
    suffix = uuid.uuid4().hex[:8]
    room = f"{job_title.strip().replace(' ','-')}-{suffix}"
    return f"https://meet.jit.si/{room}"

def send_email_with_sendgrid(to_email, candidate_name, job_title, meeting_link, ats_score):
    if not SENDGRID_API_KEY or not SENDER_EMAIL:
        raise RuntimeError("SendGrid not configured")
    
    subject = f"🎉 Interview Invitation - {job_title} Position"
    body = f"""Dear {candidate_name},

Congratulations! We are pleased to inform you that your application for the {job_title} position has been shortlisted.

Your Profile Score: {ats_score}/100

📹 Interview Meeting Link: {meeting_link}

Our recruitment team will contact you within 48 hours to schedule the interview.

Best regards,
Recruitment Team"""
    
    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )
    
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        resp = sg.send(message)
        return resp.status_code in (200, 201, 202), resp.status_code
    except Exception as e:
        return False, str(e)

# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(
    page_title="Advanced ATS Resume Shortlister", 
    layout="wide", 
    page_icon="🎯"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .score-excellent { background: #d1fae5; color: #065f46; font-weight: bold; padding: 0.5rem; border-radius: 5px; }
    .score-good { background: #dbeafe; color: #1e40af; font-weight: bold; padding: 0.5rem; border-radius: 5px; }
    .score-average { background: #fef3c7; color: #92400e; font-weight: bold; padding: 0.5rem; border-radius: 5px; }
    .score-poor { background: #fee2e2; color: #991b1b; font-weight: bold; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<div class="main-header">
    <h1>🎯 Advanced Resume Shortlister Pro</h1>
    <p>AI-Powered Intelligent Candidate Screening </p>
</div>
''', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ollama Status
    st.subheader("🤖 AI Engine")
    try:
        resp = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            st.success("✅ Ollama Connected")
            st.info(f"📦 Model: {OLLAMA_MODEL}")
        else:
            st.error("❌ Connection Error")
    except:
        st.error("❌ Ollama Not Running")
        st.code("ollama serve", language="bash")
    
    st.divider()
    
    st.subheader("📋 Job Configuration")
    job_title = st.text_input("Job Title", value="Senior Python Developer")
    job_description = st.text_area(
        "Job Description",
        height=300,
        placeholder="Enter complete job requirements, required skills, experience, qualifications..."
    )
    
    st.divider()
    
    st.subheader("🎚️ Filters")
    min_ats_score = st.slider("Minimum ATS Score", 0, 100, 65, 5)
    top_k = st.number_input("Max Resumes to Analyze", 5, 50, 15)
    
    # Clear cache button
    if st.button("🗑️ Clear Cache", help="Clear all cached data"):
        st.session_state.clear()
        st.success("✅ Cache cleared!")
    
    st.divider()
    
    run_button = st.button("🚀 Start ATS Analysis", type="primary", use_container_width=True)

# Main Content
if run_button:
    if not job_description.strip():
        st.error("❌ Please enter a job description")
    else:
        with st.spinner("🔍 Analyzing resumes with AI..."):
            df = advanced_shortlist(job_description, top_k)
        
        if df.empty:
            st.warning("⚠️ No PDF resumes found in the resumes folder")
        else:
            st.success(f"✅ Found {len(df)} resumes. Starting deep analysis...")
            
            candidates = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (_, row) in enumerate(df.iterrows()):
                status_text.text(f"📄 Analyzing: {row['filename']} ({idx+1}/{len(df)})")
                progress_bar.progress((idx + 1) / len(df))
                
                # Extract contact info with hybrid approach
                contact_info = extract_contact_info_hybrid(row['text'], row['filename'])
                
                # Calculate ATS score
                ats_data = calculate_ats_score_enhanced(
                    row['text'], 
                    job_description, 
                    row['filename']
                )
                
                # Only include if meets threshold
                if ats_data['ats_score'] >= min_ats_score:
                    candidates.append({
                        "filename": row["filename"],
                        "name": contact_info['name'],
                        "email": contact_info['email'] or "",
                        "phone": contact_info['phone'] or "",
                        "linkedin": contact_info['linkedin'] or "",
                        "similarity": round(float(row["similarity"]), 4),
                        "ats_score": ats_data['ats_score'],
                        "overall_grade": ats_data['overall_grade'],
                        "keyword_match": ats_data['keyword_match_score'],
                        "skills_match": ats_data['skills_match_score'],
                        "experience_score": ats_data['experience_score'],
                        "education_score": ats_data['education_score'],
                        "hire_recommendation": ats_data['hire_recommendation'],
                        "confidence": ats_data['confidence_level'],
                        "ats_details": ats_data,
                        "resume_text": row['text'],
                        "meeting_link": "",
                        "email_sent": False
                    })
            
            progress_bar.empty()
            status_text.empty()
            
            # Sort by ATS score
            candidates = sorted(candidates, key=lambda x: x['ats_score'], reverse=True)
            
            st.session_state["candidates"] = candidates
            st.session_state["job_description"] = job_description
            st.session_state["job_title"] = job_title
            
            if candidates:
                st.success(f"🎉 {len(candidates)} candidates qualified (Score ≥ {min_ats_score})")
            else:
                st.warning(f"⚠️ No candidates scored ≥ {min_ats_score}%. Try lowering threshold.")

# Display Results
if "candidates" in st.session_state and st.session_state["candidates"]:
    candidates = st.session_state["candidates"]
    
    # Summary
    st.subheader("📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    avg_score = sum(c['ats_score'] for c in candidates) / len(candidates)
    strong_hires = sum(1 for c in candidates if 'Strong' in c['hire_recommendation'])
    with_email = sum(1 for c in candidates if c['email'])
    
    with col1:
        st.metric("Total Qualified", len(candidates))
    with col2:
        st.metric("Average Score", f"{avg_score:.1f}%")
    with col3:
        st.metric("Strong Hires", strong_hires)
    with col4:
        st.metric("Valid Emails", with_email)
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Candidate List", "📊 Deep Analysis", "📧 Send Invitations"])
    
    with tab1:
        st.subheader("🏆 Ranked Candidates")
        
        for idx, c in enumerate(candidates):
            # Score classification
            if c['ats_score'] >= 85:
                badge = "🌟 EXCELLENT"
                score_class = "score-excellent"
            elif c['ats_score'] >= 70:
                badge = "✅ GOOD"
                score_class = "score-good"
            elif c['ats_score'] >= 60:
                badge = "⚠️ AVERAGE"
                score_class = "score-average"
            else:
                badge = "❌ WEAK"
                score_class = "score-poor"
            
            with st.expander(
                f"{badge} | #{idx+1} - {c['name'][:35]} | Score: {c['ats_score']}% | {c['overall_grade']}",
                expanded=(idx < 3)
            ):
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown(f"**📄 File:** `{c['filename']}`")
                    st.markdown(f"**👤 Name:** {c['name']}")
                    if c['email']:
                        st.markdown(f"**📧 Email:** {c['email']}")
                    else:
                        st.markdown("**📧 Email:** ❌ NOT FOUND")
                    if c['phone']:
                        st.markdown(f"**📱 Phone:** {c['phone']}")
                    if c['linkedin']:
                        st.markdown(f"**💼 LinkedIn:** {c['linkedin']}")
                
                with col2:
                    st.markdown(f"**🎯 ATS Score:** {c['ats_score']}%")
                    st.markdown(f"**🏆 Grade:** {c['overall_grade']}")
                    st.markdown(f"**✅ Recommendation:** {c['hire_recommendation']}")
                    st.markdown(f"**🎲 Confidence:** {c['confidence']}")
                
                st.markdown("**📊 Detailed Scores:**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Keywords", f"{c['keyword_match']}%")
                col2.metric("Skills", f"{c['skills_match']}%")
                col3.metric("Experience", f"{c['experience_score']}%")
                col4.metric("Education", f"{c['education_score']}%")
                
                # Show detailed ATS analysis
                ats = c['ats_details']
                
                st.markdown("---")
                st.markdown("**🎯 Matched Keywords:**")
                if ats.get('matched_keywords'):
                    st.write(", ".join(ats['matched_keywords']))
                else:
                    st.write("_None identified_")
                
                st.markdown("**❌ Missing Keywords:**")
                if ats.get('missing_keywords'):
                    st.write(", ".join(ats['missing_keywords']))
                else:
                    st.write("_None identified_")
                
                st.markdown("**💪 Key Strengths:**")
                if ats.get('key_strengths'):
                    for strength in ats['key_strengths']:
                        st.write(f"- {strength}")
                
                if ats.get('red_flags'):
                    st.markdown("**🚩 Red Flags:**")
                    for flag in ats['red_flags']:
                        st.write(f"- {flag}")
                
                if ats.get('detailed_notes'):
                    st.markdown("**📝 Detailed Notes:**")
                    st.info(ats['detailed_notes'])
    
    with tab2:
        st.subheader("📊 Deep Dive Analysis")
        
        # Select candidate
        candidate_names = [f"{c['name']} (Score: {c['ats_score']}%)" for c in candidates]
        selected = st.selectbox("Select Candidate for Deep Analysis", candidate_names)
        
        if selected:
            idx = candidate_names.index(selected)
            c = candidates[idx]
            
            # Header
            st.markdown(f"## 👤 {c['name']}")
            
            # Score visualization
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall ATS Score", f"{c['ats_score']}%", 
                          delta=f"{c['ats_score'] - avg_score:.1f}% vs avg")
            with col2:
                st.metric("Grade", c['overall_grade'])
            with col3:
                st.metric("Recommendation", c['hire_recommendation'])
            
            # Detailed breakdown
            st.markdown("---")
            st.markdown("### 📊 Score Breakdown")
            
            score_data = {
                'Category': ['Keywords', 'Skills', 'Experience', 'Education'],
                'Score': [
                    c['keyword_match'],
                    c['skills_match'],
                    c['experience_score'],
                    c['education_score']
                ]
            }
            
            # Simple bar chart representation
            for cat, score in zip(score_data['Category'], score_data['Score']):
                st.markdown(f"**{cat}:** {score}%")
                st.progress(score / 100)
            
            # Contact Information
            st.markdown("---")
            st.markdown("### 📇 Contact Information")
            
            contact_col1, contact_col2 = st.columns(2)
            with contact_col1:
                st.markdown(f"**Email:** {c['email'] if c['email'] else '❌ Not Found'}")
                st.markdown(f"**Phone:** {c['phone'] if c['phone'] else '❌ Not Found'}")
            with contact_col2:
                st.markdown(f"**LinkedIn:** {c['linkedin'] if c['linkedin'] else '❌ Not Found'}")
                st.markdown(f"**File:** `{c['filename']}`")
            
            # Skills Analysis
            st.markdown("---")
            st.markdown("### 🎯 Skills & Keywords Analysis")
            
            ats = c['ats_details']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Matched Skills:**")
                if ats.get('matched_skills'):
                    for skill in ats['matched_skills']:
                        st.success(f"✓ {skill}")
                else:
                    st.info("No skills data available")
            
            with col2:
                st.markdown("**❌ Missing Skills:**")
                if ats.get('missing_skills'):
                    for skill in ats['missing_skills']:
                        st.error(f"✗ {skill}")
                else:
                    st.info("No missing skills identified")
            
            # Experience & Education
            st.markdown("---")
            st.markdown("### 🎓 Experience & Education")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Years of Experience:** {ats.get('years_of_experience', 'Unknown')}")
            with col2:
                st.markdown(f"**Education Level:** {ats.get('education_level', 'Unknown')}")
            
            # Strengths and Red Flags
            st.markdown("---")
            st.markdown("### 💪 Strengths & 🚩 Red Flags")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Strengths:**")
                if ats.get('key_strengths'):
                    for strength in ats['key_strengths']:
                        st.success(f"✓ {strength}")
                else:
                    st.info("No strengths identified")
            
            with col2:
                st.markdown("**Red Flags:**")
                if ats.get('red_flags') and len(ats['red_flags']) > 0:
                    for flag in ats['red_flags']:
                        st.warning(f"⚠ {flag}")
                else:
                    st.success("✓ No red flags identified")
            
            # Detailed Notes
            if ats.get('detailed_notes'):
                st.markdown("---")
                st.markdown("### 📝 Detailed Analysis Notes")
                st.info(ats['detailed_notes'])
            
            # Resume Preview
            st.markdown("---")
            st.markdown("### 📄 Resume Preview")
            
            with st.expander("View Full Resume Text"):
                st.text_area("Resume Content", c['resume_text'], height=400, disabled=True)
    
    with tab3:
        st.subheader("📧 Send Interview Invitations")
        
        st.info("💡 **Tip:** Only candidates with valid email addresses can receive invitations.")
        
        # Filter candidates with email
        candidates_with_email = [c for c in candidates if c['email']]
        
        if not candidates_with_email:
            st.error("❌ No candidates have valid email addresses. Please check resume contact information.")
        else:
            st.success(f"✅ {len(candidates_with_email)} candidates have valid email addresses")
            
            # Bulk selection
            st.markdown("### 📋 Select Candidates")
            
            select_all = st.checkbox("Select All Candidates")
            
            selected_candidates = []
            
            for idx, c in enumerate(candidates_with_email):
                col1, col2, col3 = st.columns([1, 4, 2])
                
                with col1:
                    if select_all:
                        selected = st.checkbox(
                            "✓", 
                            value=True, 
                            key=f"select_{idx}",
                            label_visibility="collapsed"
                        )
                    else:
                        selected = st.checkbox(
                            "Select", 
                            key=f"select_{idx}",
                            label_visibility="collapsed"
                        )
                
                with col2:
                    st.markdown(f"**{c['name']}** | {c['email']}")
                    st.caption(f"Score: {c['ats_score']}% | {c['hire_recommendation']}")
                
                with col3:
                    if c.get('email_sent'):
                        st.success("✅ Sent")
                    else:
                        st.info("⏳ Pending")
                
                if selected:
                    selected_candidates.append(c)
            
            st.markdown("---")
            
            # Send button
            if selected_candidates:
                st.info(f"📨 Ready to send invitations to {len(selected_candidates)} candidate(s)")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button("🚀 Send Interview Invitations", type="primary", use_container_width=True):
                        if not SENDGRID_API_KEY or not SENDER_EMAIL:
                            st.error("❌ SendGrid not configured. Please set SENDGRID_API_KEY and SENDER_EMAIL in .env file")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            success_count = 0
                            failed_count = 0
                            
                            for idx, c in enumerate(selected_candidates):
                                status_text.text(f"Sending to {c['name']}... ({idx+1}/{len(selected_candidates)})")
                                
                                # Generate meeting link if not exists
                                if not c.get('meeting_link'):
                                    c['meeting_link'] = generate_jitsi_link(st.session_state.get('job_title', 'Interview'))
                                
                                # Send email
                                success, response = send_email_with_sendgrid(
                                    c['email'],
                                    c['name'],
                                    st.session_state.get('job_title', 'Position'),
                                    c['meeting_link'],
                                    c['ats_score']
                                )
                                
                                if success:
                                    c['email_sent'] = True
                                    success_count += 1
                                    st.success(f"✅ Sent to {c['name']} ({c['email']})")
                                else:
                                    failed_count += 1
                                    st.error(f"❌ Failed to send to {c['name']}: {response}")
                                
                                progress_bar.progress((idx + 1) / len(selected_candidates))
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            # Summary
                            st.balloons()
                            st.success(f"✅ Successfully sent {success_count} invitation(s)")
                            if failed_count > 0:
                                st.warning(f"⚠️ {failed_count} invitation(s) failed")
                
                with col2:
                    if st.button("🔄 Reset", use_container_width=True):
                        for c in candidates:
                            c['email_sent'] = False
                            c['meeting_link'] = ""
                        st.success("✅ Reset complete")
                        st.rerun()
            else:
                st.warning("⚠️ Please select at least one candidate to send invitations")
            
            # Meeting Links Table
            if any(c.get('meeting_link') for c in candidates_with_email):
                st.markdown("---")
                st.markdown("### 🔗 Generated Meeting Links")
                
                meeting_data = []
                for c in candidates_with_email:
                    if c.get('meeting_link'):
                        meeting_data.append({
                            "Candidate": c['name'],
                            "Email": c['email'],
                            "Meeting Link": c['meeting_link'],
                            "Status": "✅ Sent" if c.get('email_sent') else "⏳ Pending"
                        })
                
                if meeting_data:
                    df_meetings = pd.DataFrame(meeting_data)
                    st.dataframe(df_meetings, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎯 Advanced ATS Resume Shortlister Pro v2.0</p>
    <p>Powered by Sentence Transformers, Ollama AI & SendGrid</p>
    <p>Made with ❤️ for efficient recruitment</p>
</div>
""", unsafe_allow_html=True)