import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Job Recommender")

jobs = json.load(open("jobs.json"))
embeddings = np.load("job_embeddings.npy", allow_pickle=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

resume = st.text_area("Paste your resume")

domains = st.multiselect(
    "Filter by domain",
    sorted(set(j["domain"] for j in jobs))
)

if st.button("Find jobs") and resume:
    resume_emb = model.encode(resume, normalize_embeddings=True)

    idxs = [
        i for i, j in enumerate(jobs)
        if not domains or j["domain"] in domains
    ]

    sims = cosine_similarity(
        [resume_emb],
        embeddings[idxs]
    )[0]

    for i, score in sorted(zip(idxs, sims), key=lambda x: x[1], reverse=True):
        job = jobs[i]
        st.subheader(job["job_title"])
        st.write(job["organization"])
        st.write(job["domain"], "|", job["source"])
        st.write(f"Match: {score:.2f}")
        st.markdown(f"[Apply]({job['apply_link']})")
        st.divider()
