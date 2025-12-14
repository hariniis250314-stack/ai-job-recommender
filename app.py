"""
Universal AI Job Recommender
Fully flexible, no local scripts required
Embeddings are generated automatically on Streamlit Cloud
"""

import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Universal AI Job Recommender",
    page_icon="💼",
    layout="wide"
)

DATA_DIR = "data"
JOBS_PATH = os.path.join(DATA_DIR, "jobs.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "job_embeddings.npy")

# ---------------- UTILITY FUNCTIONS ----------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_jobs():
    with open(JOBS_PATH, "r") as f:
        return json.load(f)


@st.cache_resource
def load_or_create_embeddings(jobs, model):
    """
    If embeddings file exists, load it.
    Otherwise, create embeddings and save them.
    """

    if os.path.exists(EMBEDDINGS_PATH):
        return np.load(EMBEDDINGS_PATH, allow_pickle=True)

    texts = []
    for job in jobs:
        text = f"""
        Domain: {job['domain']}
        Job Title: {job['title']}
        Organization: {job['organization']}
        Description: {job['description']}
        Skills: {' '.join(job.get('skills', []))}
        """
        texts.append(text)

    embeddings = model.encode(texts, normalize_embeddings=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    return embeddings


# ---------------- LOAD EVERYTHING ----------------
model = load_model()
jobs = load_jobs()
job_embeddings = load_or_create_embeddings(jobs, model)

# ---------------- UI ----------------
st.title("💼 Universal AI Job Recommendation System")

st.markdown(
    "Upload your resume or paste text to get job recommendations "
    "across **all domains** (Software, Teaching, Government, etc.)"
)

uploaded_file = st.file_uploader(
    "Upload Resume (PDF or TXT)",
    type=["pdf", "txt"]
)

resume_text = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")

domains = st.multiselect(
    "Filter by Domain",
    sorted(set(job["domain"] for job in jobs))
)

sources = st.multiselect(
    "Filter by Source",
    sorted(set(job["source"] for job in jobs))
)

# ---------------- MATCHING ----------------
if st.button("Find Matching Jobs") and resume_text.strip():

    resume_embedding = model.encode(resume_text, normalize_embeddings=True)

    valid_indices = [
        i for i, job in enumerate(jobs)
        if (not domains or job["domain"] in domains)
        and (not sources or job["source"] in sources)
    ]

    similarities = cosine_similarity(
        [resume_embedding],
        job_embeddings[valid_indices]
    )[0]

    ranked_results = sorted(
        zip(valid_indices, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("🎯 Recommended Jobs")

    for idx, score in ranked_results[:10]:
        job = jobs[idx]

        st.markdown(f"### {job['title']}")
        st.write(job["organization"])
        st.caption(f"{job['domain']} | {job['source']} | {job['location']}")
        st.write(f"**Match Score:** {score:.2f}")
        st.markdown(f"[Apply Here]({job['apply_url']})")
        st.divider()
