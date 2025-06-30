import pandas as pd
from preprocess import load_and_prepare_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load and clean data
resume_df, jobs_df = load_and_prepare_data("data/resume_data.csv", "data/jobs.csv")

# CLEAN column names
resume_df.columns = resume_df.columns.str.strip()
resume_df.columns = resume_df.columns.str.replace('\ufeff', '', regex=True)

# Use the cleaned column
X = resume_df["skills"]
y = resume_df["job_position_name"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Save Model and Vectorizer
joblib.dump(clf, "model_training/model.pkl")
joblib.dump(vectorizer, "model_training/vectorizer.pkl")

# Print Scores
print(f"Training Score: {clf.score(X_train, y_train):.2f}")
print(f"Testing Score: {clf.score(X_test, y_test):.2f}")
