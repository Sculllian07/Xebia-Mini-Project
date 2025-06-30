import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def load_and_prepare_data(resume_path, jobs_path):
    resume_df = pd.read_csv(resume_path)
    jobs_df = pd.read_csv(jobs_path)

    # Fill missing
    for col in ['skills', 'career_objective', 'major_field_of_studies', 'role_positions']:
        resume_df[col] = resume_df[col].fillna("").apply(clean_text)
    for col in ['Key Skills', 'Job Title', 'Functional Area']:
        jobs_df[col] = jobs_df[col].fillna("").apply(clean_text)

    resume_df['full_resume_text'] = (
        resume_df['skills'] + ' ' +
        resume_df['career_objective'] + ' ' +
        resume_df['major_field_of_studies'] + ' ' +
        resume_df['role_positions']
    )

    jobs_df['full_job_text'] = (
        jobs_df['Key Skills'] + ' ' +
        jobs_df['Job Title'] + ' ' +
        jobs_df['Functional Area']
    )

    return resume_df, jobs_df
