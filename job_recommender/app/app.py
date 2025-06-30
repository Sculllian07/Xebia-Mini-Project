import streamlit as st
import joblib
import fitz  # PyMuPDF
from utils.matching import find_best_matches

st.title("Job Recommendation System")

clf = joblib.load("model_training/model.pkl")
vectorizer = joblib.load("model_training/vectorizer.pkl")

def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

uploaded_file = st.file_uploader("Upload your resume (.txt or .pdf)", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8").strip()

    if not resume_text or resume_text.startswith("Error"):
        st.warning("Could not extract usable text from the uploaded file.")
    else:
        resume_vector = vectorizer.transform([resume_text])
        prediction = clf.predict(resume_vector)[0]

        match_result = find_best_matches(resume_text, vectorizer)

        st.markdown("###  Skills Found in Your Resume:")
        if match_result["resume_skills"]:
            for skill in match_result["resume_skills"]:
                st.markdown(f"- {skill}")
        else:
            st.markdown("*No recognized technical skills found*")

        top_job = match_result["top_job"]
        st.markdown("###  Top Recommended Job Profile for You:")
        st.markdown(f"### 🔹 **{top_job['job_title']}**")
        st.markdown(f"📊 **Match Percentage:** {top_job['match_percentage']}%")

        # st.markdown("** Required Skills:**")
        # for skill in top_job["required_skills"]:
        #     st.markdown(f"- {skill}")
        # st.markdown("** Missing Skills:**")
        # if top_job["missing_skills"]:
        #     for skill in top_job["missing_skills"]:
        #         st.markdown(f"- {skill}")
        # else:
        #     st.markdown("*You meet all required skills!*")

        st.markdown("###  Other Recommended Jobs:")
        for job in match_result["other_jobs"]:
            st.markdown(f"**{job['job_title']}** — {job['match_percentage']}% match")