import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# Controlled list of real tech skills
PREDEFINED_SKILLS = [
    "Python", "Java", "C++", "C", "SQL", "JavaScript", "HTML", "CSS", "React", "Node.js",
    "Django", "Flask", "Machine Learning", "Deep Learning", "Data Analysis",
    "Data Visualization", "Pandas", "NumPy", "TensorFlow", "Keras", "AWS", "GCP", "Azure",
    "Power BI", "Tableau", "NLP", "Git", "Linux", "Agile", "Scrum", "Excel", "Bootstrap"
]

def extract_skills(text, skill_list):
    text = text.lower()
    return sorted(set(skill for skill in skill_list if skill.lower() in text))

def clean_required_skills(raw_skills):
    # Ensure required skills are extracted cleanly from key_skills column
    skills = [s.strip() for s in raw_skills.split(",")]
    return sorted(set(s for s in skills if s))

def find_best_matches(resume_text, vectorizer, top_n=4):
    jobs_df = pd.read_csv(os.path.join("data", "jobs.csv"))
    jobs_df.columns = jobs_df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "key_skills" not in jobs_df.columns or "job_title" not in jobs_df.columns:
        raise ValueError("Expected columns 'job_title' and 'key_skills' not found in jobs.csv.")

    resume_vector = vectorizer.transform([resume_text])
    job_vectors = vectorizer.transform(jobs_df["key_skills"].fillna(""))

    scores = cosine_similarity(resume_vector, job_vectors).flatten()
    jobs_df["match_score"] = scores

    top_matches = jobs_df.sort_values(by="match_score", ascending=False).head(top_n)

    resume_skills = extract_skills(resume_text, PREDEFINED_SKILLS)

    results = []
    for _, row in top_matches.iterrows():
        required = clean_required_skills(row["key_skills"])
        missing = list(set(required) - set(resume_skills))
        match_pct = round(row["match_score"] * 100, 2)
        results.append({
            "job_title": row["job_title"],
            "match_percentage": match_pct,
            "required_skills": required,
            "missing_skills": sorted(missing)
        })

    return {
        "resume_skills": resume_skills,
        "top_job": results[0],
        "other_jobs": results[1:]
    }
